import re
import json
import time
from typing import Optional
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from colorama import Fore, Style, init

# 初始化颜色输出
init(autoreset=True)

# ================= 配置区域 =================
# 如果你是本地模型 (如 Ollama/vLLM)，修改 BASE_URL 和 API_KEY
# 例如 Ollama: BASE_URL = "http://localhost:11434/v1", API_KEY = "ollama"
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 替换你的 API Key
BASE_URL = "https://api.openai.com/v1"         # 替换你的 Base URL
MODEL_NAME = "gpt-4o"                          # 替换你想测试的模型名称

# 输出文件路径
OUTPUT_FILE = "aime2024_results.jsonl"
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_model_response(problem_text: str) -> str:
    """
    发送题目给 LLM 并获取回复。
    使用了标准的 Chain-of-Thought (CoT) 提示词。
    """
    prompt = (
        "Please solve the following mathematics problem step by step.\n"
        "The final answer is a non-negative integer between 0 and 999.\n"
        "At the end of your solution, put the final integer answer inside \\boxed{}.\n\n"
        f"Problem: {problem_text}\n"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a math expert. Solve the problem carefully."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6, # 数学题通常需要较低的温度，但为了激发 CoT，0.6 是个平衡点
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"{Fore.RED}API Error: {e}{Style.RESET_ALL}")
        return ""

def extract_answer(text: str) -> Optional[int]:
    """
    从模型输出中提取 \boxed{answer} 格式的答案。
    AIME 的答案总是 000-999 的整数。
    """
    if not text:
        return None

    # 1. 优先匹配 \boxed{123} 格式
    # 兼容 \boxed{ 123 } 或 \boxed{123.} 等情况
    patterns = [
        r"\\boxed\s*\{\s*(\d+)\s*\}",
        r"\\boxed\s*\{\s*(\d+)\s*\."  # 处理偶尔带点的情况
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 取最后一个匹配项，通常是最终结论
            return int(matches[-1])
    
    # 2. 如果没有 boxed，尝试一种非常宽松的兜底策略（可选，视严格程度而定）
    # 这里我们保持严格，如果没有 boxed 视为提取失败
    return None

def main():
    print(f"{Fore.CYAN}Loading AIME 2024 dataset from Hugging Face...{Style.RESET_ALL}")
    try:
        # 加载 HuggingFaceH4/aime_2024 数据集
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    except Exception as e:
        print(f"{Fore.RED}Error loading dataset: {e}{Style.RESET_ALL}")
        return

    print(f"Loaded {len(dataset)} problems. Starting evaluation on model: {Fore.GREEN}{MODEL_NAME}{Style.RESET_ALL}\n")

    correct_count = 0
    total_count = 0
    results = []

    # 使用 tqdm 显示进度条
    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
        problem = row['problem']
        # 数据集中的答案通常是字符串，AIME 答案需转换为整数比较
        ground_truth = int(row['answer'])
        
        # 获取模型回复
        model_output = get_model_response(problem)
        
        # 提取答案
        extracted_val = extract_answer(model_output)
        
        # 判定正误
        is_correct = False
        if extracted_val is not None:
            is_correct = (extracted_val == ground_truth)
        
        if is_correct:
            correct_count += 1

        # 保存单条结果
        result_item = {
            "id": i,
            "problem": problem,
            "ground_truth": ground_truth,
            "extracted_answer": extracted_val,
            "is_correct": is_correct,
            "model_output": model_output
        }
        results.append(result_item)

        # 实时打印错误案例（可选，方便调试）
        if not is_correct:
            tqdm.write(f"{Fore.YELLOW}[Wrong] ID: {i} | Truth: {ground_truth} | Extracted: {extracted_val}{Style.RESET_ALL}")
        
        # 避免 API 速率限制，如果是免费 API 可以适当 sleep
        # time.sleep(1) 

    # 计算最终准确率
    accuracy = (correct_count / len(dataset)) * 100
    
    print("\n" + "="*50)
    print(f"Evaluation Finished for {MODEL_NAME}")
    print(f"Total Problems: {len(dataset)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {Fore.GREEN}{accuracy:.2f}%{Style.RESET_ALL}")
    print("="*50)

    # 保存详细结果到文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"Detailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()