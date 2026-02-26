from datasets import load_dataset
import re
from torch.utils.data import Dataset

class OpenThoughtsDataset(Dataset):
    """dataset for mathematical reasoning."""

    def __init__(self, data_path=None, split="train", tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        data_path = data_path or "open-thoughts/OpenThoughts3-1.2M"
        self.data = load_dataset(data_path, "main", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]

        # Extract the final numerical answer
        # Answer format: "#### 42"
        answer_number = self.extract_answer(answer)

        # Format the prompt
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"

        # ========================================================================
        # TODO 1: Tokenize the prompt
        # ========================================================================
        tokenized = self.tokenizer(prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        # END TODO 1
        # ========================================================================

        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "prompt": prompt,
            "answer": answer_number,
        }

    @staticmethod
    def extract_answer(answer_str):
        """Extract numerical answer from the answer string."""
        # GSM8K answers end with #### followed by the answer
        match = re.search(r'####\s*([-+]?\d+[\d,]*\.?\d*)', answer_str)
        if match:
            # Remove commas and convert to float
            return float(match.group(1).replace(',', ''))
        return None


class GSM8KDataset(Dataset):
    """GSM8K dataset for mathematical reasoning."""

    def __init__(self, data_path=None, split="train", tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        data_path = data_path or "./data/gsm8k"
        self.data = load_dataset(data_path, "main", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]

        # Extract the final numerical answer
        # Answer format: "#### 42"
        answer_number = self.extract_answer(answer)

        # Format the prompt
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"

        # ========================================================================
        # TODO 1: Tokenize the prompt
        # ========================================================================
        tokenized = self.tokenizer(prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        # END TODO 1
        # ========================================================================

        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "attention_mask": attention_mask.squeeze(0),
            "prompt": prompt,
            "answer": answer_number,
        }

    @staticmethod
    def extract_answer(answer_str):
        """Extract numerical answer from the answer string."""
        # GSM8K answers end with #### followed by the answer
        match = re.search(r'####\s*([-+]?\d+[\d,]*\.?\d*)', answer_str)
        if match:
            # Remove commas and convert to float
            return float(match.group(1).replace(',', ''))
        return None

def get_dataset(dataset_name, *args, **kwargs) -> Dataset:
    if dataset_name == 'gsm8k':
        return GSM8KDataset(*args, **kwargs)
    elif dataset_name == 'openthoughts':
        return OpenThoughtsDataset(*args, **kwargs)
    raise ValueError(f"Unsupported dataset {dataset_name}")