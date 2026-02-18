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

from params import DataArguments, ModelArguments, TrainingArguments
from data import OpenThoughtsDataset, get_dataset
from utils import generate_completions, compute_logprobs_from_model

# ============================================================================
# Part 2: Reward Function
# ============================================================================

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
# Part 3: GRPO Algorithm Implementation
# ============================================================================

def compute_advantages_opd(student_logprobs, teacher_logprobs):
    """
    Compute advantages using GRPO (Group Relative Policy Optimization).

    In GRPO, for each group of responses to the same prompt:
    - Advantage = reward - mean(group_rewards)

    This encourages the model to generate responses better than average.

    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        group_size: Number of responses per prompt

    Returns:
        advantages: Tensor of shape (batch_size,) containing advantages
    """
    # ========================================================================
    # TODO 2: Implement GRPO advantage computation
    # ========================================================================
    advantages = - (student_logprobs - teacher_logprobs)
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
    student_tokenizer,
    teacher_model,
    teacher_tokenizer,
    train_loader,
    optimizer,
    device,
    num_epochs=1,
    group_size=4,
    clip_eps=0.2,
    max_new_tokens=256,
):
    """
    Main GRPO training loop.

    Args:
        model: The language model
        tokenizer: The tokenizer
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        group_size: Number of samples per prompt for GRPO
        clip_eps: PPO clipping epsilon
        max_new_tokens: Maximum number of tokens to generate
    """

    rewards_list = []
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answers = batch["answer"]

            # ====================================================================
            # TODO 5: Implement the GRPO training step
            # ====================================================================
            gen_data = generate_completions(student_model, student_tokenizer, input_ids, attention_mask, max_new_tokens, num_samples=group_size)
            seq_ids, seq_mask = gen_data["output_ids"], (gen_data["output_ids"] != student_tokenizer.pad_token_id).long()
            student_logprobs = compute_logprobs_from_model(student_model, seq_ids, seq_mask)
            teacher_logprobs = compute_logprobs_from_model(teacher_model, seq_ids, seq_mask)
            advantages = compute_advantages_opd(student_logprobs, teacher_logprobs).to(device)
            loss_mask = seq_mask.clone(); loss_mask[:, :gen_data["prompt_length"]] = 0
            loss = compute_policy_loss(student_logprobs, student_logprobs.detach(), advantages, loss_mask, clip_eps) # weighted by importance sampling (always 1 if update model every step)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # END TODO 5
            # ====================================================================

            # Logging
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })
            
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
    
    return rewards_list

def load_model_and_tokenizer(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models need left padding for correct generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    return model, tokenizer

# ============================================================================
# Part 5: Main Function
# ============================================================================

def main():
    # Configuration
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    student_model_path = model_args.student_model_path
    teacher_model_path = model_args.teacher_model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2  # Small batch size for homework
    group_size = 8  # Number of samples per prompt
    num_epochs = 5
    learning_rate = 5e-6
    max_new_tokens = 128

    print("Loading student tokenizer and model...")
    student_model, student_tokenizer = load_model_and_tokenizer(student_model_path, device)
    print("Loading teacher tokenizer and model...")
    teacher_model, teacher_tokenizer = load_model_and_tokenizer(teacher_model_path, device)

    print("Loading dataset...")
    train_dataset = get_dataset(data_args.data_path, split="train[:100]", tokenizer=student_tokenizer)  # Use small subset
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    print("Starting GRPO training...")
    rewards_list = train_opd(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
    )

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)

    ckpt_dir = "./saved_model"
    student_model.save_pretrained(ckpt_dir)
    student_tokenizer.save_pretrained(ckpt_dir)
    print(f"The model has been saved to {ckpt_dir}")

    # Plot training curves
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
    plt.savefig('grpo_training_curves.png')
    print(f"Training curves saved to grpo_training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
