import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_logprobs_from_model(model, input_ids, attention_mask):
    """
    Compute log probabilities for given sequences.

    Args:
        model: The language model
        input_ids: Input token ids, shape (batch_size, seq_len)
        attention_mask: Attention mask, shape (batch_size, seq_len)

    Returns:
        logprobs: Log probabilities, shape (batch_size, seq_len)
    """
    # ========================================================================
    # TODO 4: Compute log probabilities
    # ========================================================================
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Predict next token
    target_ids = input_ids[:, 1:]       # Ground truth next token
    token_logprobs = logits.log_softmax(dim=-1).gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    logprobs = F.pad(token_logprobs, (0, 1), value=0.0)
    # END TODO 4
    # ========================================================================

    return logprobs

def compute_full_logprobs_from_model(model, input_ids, attention_mask):
    """
    Compute log probabilities distribution for given sequences.

    Args:
        model: The language model
        input_ids: Input token ids, shape (batch_size, seq_len)
        attention_mask: Attention mask, shape (batch_size, seq_len)

    Returns:
        logprobs: Log probabilities distribution, shape (batch_size, seq_len, vocab_size)
    """

    outputs = model(input_ids, attention_mask=attention_mask)
    
    logprobs = outputs.logits.log_softmax(dim=-1)

    return logprobs

def generate_completions(model, tokenizer, input_ids, attention_mask,
                        max_new_tokens=256, temperature=1.0, num_samples=4):
    """
    Generate multiple completions for each prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token ids, shape (batch_size, seq_len)
        attention_mask: Attention mask, shape (batch_size, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        num_samples: Number of samples per prompt

    Returns:
        all_outputs: Dictionary containing generated sequences and info
    """
    model.eval()

    # Repeat inputs for multiple samples
    batch_size = input_ids.shape[0]
    input_ids_repeated = input_ids.repeat_interleave(num_samples, dim=0)
    attention_mask_repeated = attention_mask.repeat_interleave(num_samples, dim=0)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask_repeated,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode completions
    completions = []
    for output in outputs:
        # Get only the generated part (exclude prompt)
        prompt_len = input_ids.shape[1]
        generated_ids = output[prompt_len:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return {
        "output_ids": outputs,
        "completions": completions,
        "prompt_length": input_ids.shape[1],
    }

