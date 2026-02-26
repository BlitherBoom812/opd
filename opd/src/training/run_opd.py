import functools
import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from tqdm import tqdm
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.distributed.elastic.multiprocessing.errors import record
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    ModuleWrapPolicy, 
)
from opd.src.utils.params import DataArguments, ModelArguments, TrainingArguments
from opd.src.data.dataset import get_dataset
from opd.src.utils import generate_completions, compute_logprobs_from_model

torch.manual_seed(42)
# ============================================================================
# Part 2: Reward Function
# ============================================================================
local_rank = None

def print_main_process(*args, **kwargs):
   if local_rank == 0:
       print(*args, **kwargs)

def extract_answer_from_completion(completion):
    """Extract the final answer from model's completion."""
    # Look for patterns like "The answer is 42" or "#### 42"
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

    Args:
        completions: List of completion strings
        ground_truth_answers: List of ground truth answers

    Returns:
        rewards: Tensor of shape (batch_size,) with values 0 or 1
    """
    rewards = []
    for completion, gt_answer in zip(completions, ground_truth_answers):
        predicted = extract_answer_from_completion(completion)
        if predicted is not None and abs(predicted - gt_answer) < 1e-3:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards, dtype=torch.float32)


# ============================================================================
# Part 3: OPD Algorithm Implementation
# ============================================================================

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


def compute_policy_loss(logprobs, old_logprobs, advantages, loss_mask, clip_eps=0.2):
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
# Part 4: Training Loop
# ============================================================================

def train_opd(
    student_model,
    student_tokenizer: AutoTokenizer,
    ref_model,
    teacher_model,
    teacher_tokenizer,
    device,
    train_loader: DataLoader,
    optimizer,
    local_rank,
    num_epochs=1,
    group_size=4,
    clip_eps=0.2,
    max_new_tokens=256,
):
    """
    Main OPD training loop.

    Args:
        model: The language model
        tokenizer: The tokenizer
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        group_size: Number of samples per prompt for OPD
        clip_eps: PPO clipping epsilon
        max_new_tokens: Maximum number of tokens to generate
    """

    rewards_list = []
    for epoch in range(num_epochs):
        print_main_process(f"\n{'='*60}")
        print_main_process(f"Epoch {epoch + 1}/{num_epochs}")
        print_main_process(f"{'='*60}")
        train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training", disable=local_rank != 0)

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"]
            print_main_process(f"prompt: {batch['prompt'][0]}")
            # ====================================================================
            # TODO 5: Implement the OPD training step
            # ====================================================================
            # For FSDP, we need to properly unwrap the model for generation
            with FSDP.summon_full_params(student_model, writeback=False):
                gen_data = generate_completions(student_model, student_tokenizer, input_ids, attention_mask, max_new_tokens, num_samples=group_size, synced_gpus=True)
            seq_ids, seq_mask = gen_data["output_ids"], (gen_data["output_ids"] != student_tokenizer.pad_token_id).long()
            seq_ids = seq_ids.to(device)

            print_main_process(f"generated data sample: {seq_ids.shape}\n{seq_ids[0, :]}\n{gen_data['completions'][0]}")
            student_logprobs = compute_logprobs_from_model(student_model, seq_ids, seq_mask)
            print_main_process(f"student logprobs: {student_logprobs.shape}, {student_logprobs}")
            with torch.no_grad():
                ref_logprobs = compute_logprobs_from_model(ref_model, seq_ids, seq_mask)
                print_main_process(f"ref logprobs: {ref_logprobs.shape}, {ref_logprobs}")
                teacher_logprobs = compute_logprobs_from_model(teacher_model, seq_ids, seq_mask)
                print_main_process(f"teacher logprobs: {teacher_logprobs.shape}, {teacher_logprobs}")
            
            advantages = compute_advantages_opd(student_logprobs.detach(), ref_logprobs.detach(), teacher_logprobs.detach()).to(device)
            print_main_process(f"advantages: {advantages.shape}, {advantages}")
            loss_mask: torch.Tensor = seq_mask.clone(); loss_mask[:, :gen_data["prompt_length"]] = 0

            loss = compute_policy_loss(student_logprobs, student_logprobs.detach(), advantages, loss_mask, clip_eps) # weighted by importance sampling (always 1 if update model every step)
            # loss = policy_loss + kl_loss
            # print_main_process(f"total loss: {loss}, policy loss: {policy_loss}, kl loss: {kl_loss}")
            print_main_process(f"loss: {loss}")

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # END TODO 5
            # ====================================================================

            # Logging
            loss_tensor = loss.detach().clone()
            # Gather losses from all processes
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_losses, loss_tensor)
            if local_rank == 0:
                all_losses_tensor = torch.stack(gathered_losses)
                print_main_process(f"all_losses: {all_losses_tensor.shape}, {all_losses_tensor}")
                total_loss += all_losses_tensor.mean().item()
            num_batches += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })
            
        avg_loss = total_loss / num_batches
        print_main_process(f"\nEpoch {epoch + 1} Summary:")
        print_main_process(f"  Average Loss: {avg_loss:.4f}")
    
    return rewards_list

def load_model_and_tokenizer(model_path, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models need left padding for correct generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device  # Load on CPU first, FSDP will handle sharding
    )
    return model, tokenizer

# ============================================================================
# Part 5: Main Function
# ============================================================================
def get_transformer_layer_cls(model):
    """
    智能获取模型的Transformer层类（无需手动import）
    """
    # 1. 尝试从 Hugging Face 模型属性中获取 (最准确)
    # 处理 PEFT/LoRA 模型的情况，需访问 base_model
    target_model = getattr(model, "base_model", model)
    
    if hasattr(target_model, "_no_split_modules") and target_model._no_split_modules:
        no_split_modules = set(target_model._no_split_modules)
        layer_cls = set()
        
        # 遍历模型所有模块，找到名字匹配的 Class
        for module in target_model.modules():
            if module.__class__.__name__ in no_split_modules:
                layer_cls.add(module.__class__)
        
        if layer_cls:
            return layer_cls
    # 2. 如果没有 _no_split_modules，尝试启发式查找 (Fallback)
    # 查找常见的层命名习惯，如 'layers', 'h', 'blocks'
    for attr_name in ["layers", "h", "blocks", "model.layers"]:
        modules = dict(target_model.named_modules())
        if attr_name in modules and len(modules[attr_name]) > 0:
            # 获取第一个Layer的类型
            return {type(modules[attr_name][0])}
    return set()
def setup_fsdp_model(model, local_rank):
    """
    Setup FSDP wrapping for the model (Dynamic & Smart)
    """
    
    # 动态获取当前模型的 Transformer Block Class
    transformer_layer_cls = get_transformer_layer_cls(model)
    if transformer_layer_cls:
        # 使用 ModuleWrapPolicy (更现代的写法)
        # 如果是旧版 PyTorch，可以使用 transformer_auto_wrap_policy + functools.partial
        auto_wrap_policy = ModuleWrapPolicy(transformer_layer_cls)
        print_main_process(f"FSDP: Applied Transformer wrapping for {transformer_layer_cls}")
    else:
        print_main_process("FSDP: Warning! specific layer class not found. Falling back to size-based wrapping.")
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=10_000_000  # 建议调大一点，1M太小了，通常切分Block至少是10M+
        )
    # Mixed precision settings
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    # Wrap the model
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True, # 强烈建议开启，兼容 compile 和 PEFT
        sync_module_states=True
    )
    return fsdp_model

@record
def main():
    global local_rank
    # Initialize distributed training
    init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    print(f"local rank: {repr(local_rank)}, {type(local_rank)}")

    # Configuration
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    batch_size = 2  # Small batch size for homework
    group_size = 8  # Number of samples per prompt
    num_epochs = 5
    learning_rate = 5e-6
    max_new_tokens = 128

    world_size = int(os.environ['WORLD_SIZE'])
    print_main_process(f"Training on {world_size} GPUs")

    cpu_device = "cpu" # load on cpu first, then handled by fsdp
    student_model_path = model_args.student_model_path
    teacher_model_path = model_args.teacher_model_path
    
    print_main_process("Loading student tokenizer and model...")
    student_model, student_tokenizer = load_model_and_tokenizer(student_model_path, cpu_device)
    
    print_main_process("Loading ref tokenizer and model...")
    ref_model, _ = load_model_and_tokenizer(student_model_path, cpu_device)
    ref_model.requires_grad_(False)
    
    print_main_process("Loading teacher tokenizer and model...")
    teacher_model, teacher_tokenizer = load_model_and_tokenizer(teacher_model_path, cpu_device)
    teacher_model.requires_grad_(False)

    print_main_process("Setting up FSDP model and optimizer...")
    # Wrap model with FSDP
    student_model = setup_fsdp_model(student_model, local_rank)
    teacher_model = setup_fsdp_model(teacher_model, local_rank)
    ref_model = setup_fsdp_model(ref_model, local_rank)

    ref_model.eval()
    teacher_model.eval()

    print_main_process("Loading dataset...")
    train_dataset = get_dataset(data_args.dataset_name, data_path=data_args.data_path, split="train[:100]", tokenizer=student_tokenizer)  # Use small subset
    print_main_process(f"Dataset Loaded, total length: {len(train_dataset)}")

    # Use DistributedSampler for proper data distribution
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda x: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in x],
                batch_first=True,
                padding_value=student_tokenizer.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in x],
                batch_first=True,
                padding_value=0
            ),
            "prompt": [item["prompt"] for item in x],
            "answer": [item["answer"] for item in x],
        }
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    print_main_process("Starting OPD training...")
    rewards_list = train_opd(
        student_model=student_model,
        ref_model=ref_model,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        local_rank=local_rank,
        num_epochs=num_epochs,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
    )

    # Ensure all processes finish training
    torch.distributed.barrier()
    print_main_process("\n" + "="*60)
    print_main_process("Training completed!")
    print_main_process("="*60)

    # Get full model state dict from FSDP
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    # 所有卡共同参与收集权重
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    full_state_dict = get_model_state_dict(student_model, options=options)
    # 只在 Rank 0 负责落盘保存
    if local_rank == 0:
        ckpt_dir = training_args.output_dir
        torch.save(full_state_dict, f"{ckpt_dir}/pytorch_model.bin")
        student_tokenizer.save_pretrained(ckpt_dir)
        print_main_process(f"The model has been saved to {ckpt_dir}")

    # Plot training curves (only on main process)
    if local_rank == 0:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import uniform_filter1d

        # Smooth the curves using a moving average
        def smooth_curve(data, window_size=10):
            if len(data) < window_size:
                return data
            return uniform_filter1d(data, size=window_size, mode='nearest')

        plt.figure(figsize=(12, 5))
        # Plot rewards
        smoothed_rewards = smooth_curve(rewards_list)
        plt.plot(rewards_list, alpha=0.3, label='Raw')
        plt.plot(smoothed_rewards, label='Smoothed')
        plt.xlabel('Training Step')
        plt.ylabel('Average Reward')
        plt.title('Average Reward over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig('OPD_training_curves.png')
        print_main_process(f"Training curves saved to OPD_training_curves.png")
        plt.show()

    # Cleanup distributed training
    destroy_process_group()


if __name__ == "__main__":
    main()
