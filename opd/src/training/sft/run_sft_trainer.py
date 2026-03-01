import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
)
import datasets
import transformers
from opd.src.utils.params import DataArguments, ModelArguments, TrainingArguments

torch.manual_seed(42)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# SFT Data Collator
# ============================================================================

class SFTDataCollator:
    """Custom data collator for SFT training."""

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

        # For SFT, labels are the same as input_ids (shifted internally by the model)
        batch["labels"] = batch["input_ids"].clone()

        return batch


# ============================================================================
# SFT Trainer Implementation
# ============================================================================

class SFTTrainer(Trainer):
    """Custom Trainer implementing SFT (Supervised Fine-Tuning)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
    #     """
    #     Compute SFT loss (standard language modeling loss).
    #     """
    #     # Move inputs to device
    #     input_ids = inputs["input_ids"].to(model.device)
    #     attention_mask = inputs["attention_mask"].to(model.device)
    #     labels = inputs["labels"].to(model.device)

    #     # Forward pass
    #     outputs = model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #     )

    #     loss = outputs.loss

    #     return (loss, outputs) if return_outputs else loss


# ============================================================================
# Model Loading and Setup
# ============================================================================

def load_model_and_tokenizer(model_path, device=None):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT typically uses right padding

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model, tokenizer


def print_main_process(*args, **kwargs):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        logger.warning(" ".join([str(arg) for arg in args]), **kwargs)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training function using Trainer API."""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Load model and tokenizer
    model_path = model_args.student_model_path
    print_main_process(f"Loading tokenizer and model from {model_path}...")
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load dataset
    print_main_process("Loading dataset...")
    from opd.src.data.dataset import get_dataset
    train_dataset = get_dataset(
        data_args.dataset_name,
        data_path=data_args.data_path,
        split=data_args.split,
        tokenizer=tokenizer,
        max_length=training_args.max_seq_length,
        load_from_disk=True, # TODO: remove hard code
    )
    print_main_process(f"Dataset loaded, total length: {len(train_dataset)}")

    # Setup data collator
    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_seq_length
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Start training
    print_main_process("Starting SFT training...")
    train_result = trainer.train()

    # Save model
    print_main_process("Training completed!")
    trainer.save_model()
    trainer.save_metrics("train", train_result.metrics)


if __name__ == "__main__":
    main()
