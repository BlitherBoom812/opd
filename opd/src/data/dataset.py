import datasets
import re
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class OpenThoughtsDataset(Dataset):
    """OpenThoughts dataset for reasoning tasks."""

    def __init__(self, data_path=None, split="train", tokenizer=None, max_length=512, load_from_disk=False):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        assert self.tokenizer, "tokenizer should not be None"
        self.max_length = max_length
        data_path = data_path or "open-thoughts/OpenThoughts3-1.2M"
        # OpenThoughts doesn't have "main" config, load directly
        if load_from_disk:
            self.data = datasets.load_from_disk(data_path)
        else:
            self.data = datasets.load_dataset(data_path, split=split)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        # 1. 提取 human 和 gpt 的对话内容
        conversations = item["conversations"]
        prompt_text = ""
        response_text = ""
        for conv in conversations:
            if conv.get("from") == "human":
                prompt_text = conv.get("value", "")
            elif conv.get("from") == "gpt":
                response_text = conv.get("value", "")
        # 2. 转换为符合 Chat Template 规范的 messages 格式
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text}
        ]
        
        # 3. 使用 apply_chat_template 获取文本
        # full_text 包含：系统提示词 + 用户问题 + Assistant回答 + 结束符(<|im_end|>)
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False
        )
        
        # 获取只包含 Prompt 的文本，用来计算需要屏蔽的长度
        # add_generation_prompt=True 会在末尾自动补上 "<|im_start|>assistant\n"，这是模型开始回答前的确切边界
        prompt_messages = [{"role": "user", "content": prompt_text}]
        prompt_text_only = self.tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        # 4. 对完整文本进行 Tokenize（填充并截断）
        tokenized_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            # padding="max_length", # 在 collate_fn 里做动态填充是更加推荐的做法
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized_full["input_ids"].squeeze(0)
        attention_mask = tokenized_full["attention_mask"].squeeze(0)
        # 对 Prompt 文本进行 Tokenize（只为了获取长度，不需要 padding）
        tokenized_prompt = self.tokenizer(
            prompt_text_only,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        prompt_len = tokenized_prompt["input_ids"].shape[1]
        # 5. 构建 labels (重点：只计算 Answer 的 Loss)
        labels = input_ids.clone()
        
        # 将 Prompt 部分的 label 全部替换为 -100
        # 如果 prompt_len 超过了 max_length，切片操作 [:prompt_len] 依然安全
        labels[:prompt_len] = -100
        
        # 将 Padding 填充部分的 label 也替换为 -100
        labels[attention_mask == 0] = -100
        # 提取数值型答案用于 reward (如果业务需要)
        answer_number = self.extract_answer(response_text) if response_text else None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt": prompt_text,
            "answer": response_text,
            "answer_number": answer_number,
        }
    
    @staticmethod
    def extract_answer(answer_str):
        """Extract numerical answer from the answer string."""
        if not isinstance(answer_str, str):
            return None
        # Try common patterns for numerical answers
        patterns = [
            r'####\s*([-+]?\d+[\d,]*\.?\d*)',  # GSM8K format
            r'(?:answer|result)\s*(?:is|=|:)\s*([-+]?\d+[\d,]*\.?\d*)',  # "answer is 42" or "result: 42"
            r'\$([-+]?\d+[\d,]*\.?\d*)\$',  # LaTeX format: $42$
        ]
        for pattern in patterns:
            match = re.search(pattern, answer_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(',', ''))
                except:
                    continue
        return None


class GSM8KDataset(Dataset):
    """GSM8K dataset for mathematical reasoning."""

    def __init__(self, data_path=None, split="train", tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        assert self.tokenizer, "tokenizer should not be None"
        self.max_length = max_length
        data_path = data_path or "./data/gsm8k"
        self.data = datasets.load_dataset(data_path, "main", split=split)

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
    elif dataset_name == 'open-thoughts':
        return OpenThoughtsDataset(*args, **kwargs)
    raise ValueError(f"Unsupported dataset {dataset_name}")

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/mnt/nvme0/tsz/modelscope_cache/models/Qwen/Qwen3-0.6B')
    dataset = get_dataset('open-thoughts', data_path="data/openthoughts_math_w_answer", split='train', tokenizer=tokenizer, load_from_disk=True, max_length=32768)
    print(dataset)
    breakpoint()
