import functools
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.distributed.elastic.multiprocessing.errors import record
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
from opd.src.utils import get_optimizer_grouped_parameters

torch.manual_seed(42)

# ============================================================================
# Part 1: Helper Functions
# ============================================================================
local_rank = None

def print_main_process(*args, **kwargs):
   if local_rank == 0:
       print(*args, **kwargs)


# ============================================================================
# Part 2: SFT Training Loop
# ============================================================================

def train_sft(
    model,
    tokenizer: AutoTokenizer,
    device,
    train_loader: DataLoader,
    optimizer,
    local_rank,
    num_epochs=1,
):
    """
    Main SFT training loop.

    Args:
        model: The language model
        tokenizer: The tokenizer
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        local_rank: Local rank for distributed training
        num_epochs: Number of training epochs
    """

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
            labels = input_ids.clone()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            loss_tensor = loss.detach().clone()
            # Gather losses from all processes
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_losses, loss_tensor)
            if local_rank == 0:
                all_losses_tensor = torch.stack(gathered_losses)
                print_main_process(f"loss: {all_losses_tensor.mean().item()}, all_losses: {all_losses_tensor.shape}, {all_losses_tensor}")
                total_loss += all_losses_tensor.mean().item()
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })

        avg_loss = total_loss / num_batches
        print_main_process(f"\nEpoch {epoch + 1} Summary:")
        print_main_process(f"  Average Loss: {avg_loss:.4f}")


def load_model_and_tokenizer(model_path, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # SFT typically uses right padding
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    )
    return model, tokenizer


# ============================================================================
# Part 3: FSDP Setup
# ============================================================================

def get_transformer_layer_cls(model):
    """
    智能获取模型的Transformer层类（无需手动import）
    """
    target_model = getattr(model, "base_model", model)

    if hasattr(target_model, "_no_split_modules") and target_model._no_split_modules:
        no_split_modules = set(target_model._no_split_modules)
        layer_cls = set()

        for module in target_model.modules():
            if module.__class__.__name__ in no_split_modules:
                layer_cls.add(module.__class__)

        if layer_cls:
            return layer_cls

    for attr_name in ["layers", "h", "blocks", "model.layers"]:
        modules = dict(target_model.named_modules())
        if attr_name in modules and len(modules[attr_name]) > 0:
            return {type(modules[attr_name][0])}
    return set()


def setup_fsdp_model(model, local_rank):
    """
    Setup FSDP wrapping for the model
    """
    transformer_layer_cls = get_transformer_layer_cls(model)
    if transformer_layer_cls:
        auto_wrap_policy = ModuleWrapPolicy(transformer_layer_cls)
        print_main_process(f"FSDP: Applied Transformer wrapping for {transformer_layer_cls}")
    else:
        print_main_process("FSDP: Warning! specific layer class not found. Falling back to size-based wrapping.")
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=10_000_000
        )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True,
        sync_module_states=True
    )
    return fsdp_model


def setup_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            device_id=device
        )

    return device, local_rank, global_rank


# ============================================================================
# Part 4: Main Function
# ============================================================================

@record
def main():
    global local_rank
    # Initialize distributed training
    device, local_rank, global_rank = setup_distributed()

    print(f"local rank: {repr(local_rank)}, {type(local_rank)}")

    # Configuration
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    world_size = int(os.environ['WORLD_SIZE'])
    print_main_process(f"Training on {world_size} GPUs")

    cpu_device = "cpu"
    model_path = model_args.student_model_path

    print_main_process(f"Loading tokenizer and model from {model_path}...")
    model, tokenizer = load_model_and_tokenizer(model_path, cpu_device)

    print_main_process("Setting up FSDP model and optimizer...")
    # Wrap model with FSDP
    model = setup_fsdp_model(model, local_rank)

    print_main_process("Loading dataset...")
    train_dataset = get_dataset(
        data_args.dataset_name,
        data_path=data_args.data_path,
        split=data_args.split,
        tokenizer=tokenizer,
        max_length=training_args.max_seq_length,
        load_from_disk=True, # TODO: remove hard code
    )
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
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=lambda x: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in x],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in x],
                batch_first=True,
                padding_value=0
            ),
        }
    )

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, training_args.weight_decay)
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    print_main_process("Starting SFT training...")
    train_sft(
        model=model,
        tokenizer=tokenizer,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        local_rank=local_rank,
        num_epochs=int(training_args.num_train_epochs),
    )

    # Ensure all processes finish training
    torch.distributed.barrier()
    print_main_process("\n" + "="*60)
    print_main_process("Training completed!")
    print_main_process("="*60)

    # Get full model state dict from FSDP
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    full_state_dict = get_model_state_dict(model, options=options)

    # Only save on Rank 0
    if local_rank == 0:
        ckpt_dir = training_args.output_dir
        torch.save(full_state_dict, f"{ckpt_dir}/pytorch_model.bin")
        tokenizer.save_pretrained(ckpt_dir)
        print_main_process(f"The model has been saved to {ckpt_dir}")

    # Cleanup distributed training
    destroy_process_group()


if __name__ == "__main__":
    main()
