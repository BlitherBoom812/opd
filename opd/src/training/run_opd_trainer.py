import functools
import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import logging
import datasets
import transformers
from opd.src.utils.params import DataArguments, ModelArguments, TrainingArguments
from opd.src.data.dataset import get_dataset
from opd.src.utils import generate_completions, compute_logprobs_from_model
import torch.distributed as dist

torch.manual_seed(42)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    # level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================================
# OPD Reward Function
# ============================================================================

def extract_answer_from_completion(completion):
    """Extract the final answer from model's completion."""
    patterns = [
        r'answer is\s*([-+]?\d+[\d,]*\.?\d*)',
        r'####\s*([-+]?\d+[\d,]*\.?\d*)',
        r'=\s*([-+]?\d+[\d,]*\.?\d*)(?:\s|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))
    return None


def compute_reward(completions, ground_truth_answers):
    """
    Compute binary rewards: 1 if correct, 0 if incorrect.
    """
    rewards = []
    for completion, gt_answer in zip(completions, ground_truth_answers):
        predicted = extract_answer_from_completion(completion)
        if predicted is not None and abs(predicted - gt_answer) < 1e-3:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def compute_advantages_opd(student_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor):
    """
    Compute advantages using OPD

    In OPD, for each group of responses to the same prompt:
    - Advantage = reward - mean(group_rewards)

    This encourages the model to generate responses better than average.

    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        group_size: Number of responses per prompt

    Returns:
        advantages: Tensor of shape (batch_size,) containing advantages
    """
    # ========================================================================
    # TODO 2: Implement OPD advantage computation
    # ========================================================================
    reverse_kl = (student_logprobs - teacher_logprobs) + (student_logprobs - ref_logprobs)
    advantages = -reverse_kl
    # END TODO 2
    # ========================================================================

    return advantages


def compute_policy_loss(logprobs, old_logprobs, advantages, loss_mask, clip_eps=0.2) -> torch.Tensor:
    """
    Compute PPO-style clipped policy loss.

    Args:
        logprobs: Current policy log probabilities, shape (batch_size, seq_len)
        old_logprobs: Old policy log probabilities, shape (batch_size, seq_len)
        advantages: Advantages, shape (batch_size,)
        loss_mask: Mask for valid tokens, shape (batch_size, seq_len)
        clip_eps: Clipping epsilon for PPO

    Returns:
        loss: Scalar loss value
    """
    # ========================================================================
    # TODO 3: Implement PPO-style policy loss
    # ========================================================================
    ratio = torch.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages.unsqueeze(1)
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.unsqueeze(1)
    loss = -torch.min(surr1, surr2)
    loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    # END TODO 3
    # ========================================================================

    return loss

# ============================================================================
# OPD Trainer Implementation
# ============================================================================

class OPDTrainingArguments(TrainingArguments):
    def __init__(
        self,
        group_size: int = 8,
        max_new_tokens: int = 128,
        kl_weight: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.kl_weight = kl_weight


class OPDDataCollator:
    """Custom data collator for OPD training."""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch = {}

        # Pad input_ids and attention_mask
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        batch["prompt"] = [f["prompt"] for f in features]
        batch["answer"] = [f["answer"] for f in features]

        return batch


class OPDTrainer(Trainer):
    """Custom Trainer implementing OPD algorithm."""

    def __init__(
        self,
        student_tokenizer: AutoTokenizer,
        ref_model,
        teacher_model,
        teacher_tokenizer,
        local_rank,
        num_epochs=1,
        group_size=4,
        clip_eps=0.2,
        max_new_tokens=256,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        print_main_process(f"training model device: {self.model.device}")
        self.student_model = self.model
        self.student_tokenizer = student_tokenizer
        self.ref_model = ref_model
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.local_rank = local_rank
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.max_new_tokens = max_new_tokens

        # Freeze teacher and reference models
        if self.teacher_model is not None:
            self.teacher_model.to(self.args.device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        print_main_process(f"teacher model device: {teacher_model.device}")

        if self.ref_model is not None:
            self.ref_model.to(self.args.device)
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        print_main_process(f"ref model device: {ref_model.device}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Compute OPD loss.
        """
        # Move inputs to device
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        answers = inputs["answer"]

        # Generate completions using the student model
        with FSDP.summon_full_params(model, writeback=False):
            gen_data = generate_completions(
                model,
                self.student_tokenizer,
                input_ids,
                attention_mask,
                self.max_new_tokens,
                num_samples=self.group_size,
                synced_gpus=True
            )

        seq_ids = gen_data["output_ids"].to(model.device)
        seq_mask = (seq_ids != self.student_tokenizer.pad_token_id).long().to(model.device)
        print_main_process(f"generated data sample: {seq_ids.shape}\n{seq_ids[0, :]}\n{gen_data['completions'][0]}")

        # Compute log probabilities
        student_logprobs = compute_logprobs_from_model(model, seq_ids, seq_mask)
        print_main_process(f"student logprobs: {student_logprobs.shape}, {student_logprobs}")

        with torch.no_grad():
            print_main_process("ref model device: ", self.ref_model.device)
            print_main_process("ref input device: ", seq_ids.device, seq_mask.device)
            ref_logprobs = compute_logprobs_from_model(self.ref_model, seq_ids, seq_mask)
            print_main_process(f"ref logprobs: {ref_logprobs.shape}, {ref_logprobs}")
            
            print_main_process("teacher model device: ", self.teacher_model.device)
            print_main_process("ref input device: ", seq_ids.device, seq_mask.device)
            teacher_logprobs = compute_logprobs_from_model(self.teacher_model, seq_ids, seq_mask)
            print_main_process(f"teacher logprobs: {teacher_logprobs.shape}, {teacher_logprobs}")

        # Compute OPD advantages
        advantages = compute_advantages_opd(student_logprobs.detach(), ref_logprobs.detach(), teacher_logprobs.detach())
        print_main_process(f"advantages: {advantages.shape}, {advantages}")

        # Create loss mask
        loss_mask = seq_mask.clone()
        loss_mask[:, :gen_data["prompt_length"]] = 0

        # Compute policy loss (PPO-style)
        total_loss = compute_policy_loss(student_logprobs, student_logprobs.detach(), advantages, loss_mask, self.clip_eps) # weighted by importance sampling (always 1 if update model every step)

        return (total_loss, {
            "total_loss": total_loss.detach(),
            "advantages": advantages.detach(),
            "student_logprobs": student_logprobs.detach(),
            "teacher_logprobs": teacher_logprobs.detach(),
        }) if return_outputs else total_loss

    def get_train_dataloader(self):
        """Override to use custom data collator."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        return self.accelerator.prepare(dataloader)

class OPDCallback(TrainerCallback):
    """Callback for OPD-specific logging."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log OPD-specific metrics."""
        if logs is None:
            return

        # Log additional OPD metrics if available
        if "total_loss" in logs:
            logs["opd_total_loss"] = logs["policy_loss"]

# ============================================================================
# Model Loading and Setup
# ============================================================================

def load_model_and_tokenizer(model_path, device="cpu"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map=device
    )
    return model, tokenizer

local_rank = None
def print_main_process(*args, **kwargs):
   if local_rank == 0:
        logger.warning(" ".join([str(arg) for arg in args]), **kwargs)
#         print(*args, **kwargs)

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training function using Trainer API."""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, OPDTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    group_size = 8  # Number of samples per prompt
    max_new_tokens = 128

    global local_rank
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"local rank: {repr(local_rank)}, {type(local_rank)}")

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Load models and tokenizers
    initial_device = None

    print_main_process("Loading student model and tokenizer...")
    student_model, student_tokenizer = load_model_and_tokenizer(
        model_args.student_model_path, initial_device
    )

    print_main_process("Loading reference model...")
    ref_model, _ = load_model_and_tokenizer(
        model_args.student_model_path, initial_device
    )

    print_main_process("Loading teacher model...")
    teacher_model, teacher_tokenizer = load_model_and_tokenizer(
        model_args.teacher_model_path, initial_device
    )

    # Load dataset
    print_main_process("Loading dataset...")

    train_dataset = get_dataset(
        data_args.dataset_name,
        data_path=data_args.data_path,
        split="train[:100]",
        tokenizer=student_tokenizer
    )

    print_main_process(f"Dataset loaded, total length: {len(train_dataset)}")

    # Setup data collator
    data_collator = OPDDataCollator(
        tokenizer=student_tokenizer,
        max_length=training_args.max_seq_length
    )

    # Initialize trainer
    trainer = OPDTrainer(
        student_tokenizer=student_tokenizer,
        ref_model=ref_model,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        local_rank=local_rank,
        num_epochs=training_args.num_train_epochs,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        model=student_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Add callback
    trainer.add_callback(OPDCallback)

    # Start training
    print_main_process("Starting OPD training...")

    train_result = trainer.train()

    # Save model
    print_main_process("Training completed!")
    if local_rank == 0:
        trainer.save_model()
        trainer.save_metrics("train", train_result.metrics)

if __name__ == "__main__":
    main()