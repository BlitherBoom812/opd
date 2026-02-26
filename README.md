# On-Policy Distillation Training

## Overview

OPD is a distillation approach that trains a student model to improve upon a teacher model by:
- Using a reference model for stability
- Computing advantages based on reverse KL divergence
- Applying PPO-style clipped policy loss
- Supporting distributed training with FSDP (Fully Sharded Data Parallel)

## Prerequisites

- Python 3.10+
- PyTorch 2.10.0+
- Multiple GPUs (recommended for distributed training)
- CUDA-compatible GPUs

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Supported Datasets

The code supports two main datasets:

1. **GSM8K** - Grade school math problems
2. **OpenThoughts** - Mathematical reasoning dataset

### Preparing GSM8K Dataset

1. Download the GSM8K dataset:
```bash
# Option 1: Using Hugging Face datasets (automatically downloaded)
# The dataset will be downloaded automatically when you run the training

# Option 2: Manual download and local storage
# Create a data directory
mkdir -p ./data/gsm8k

# The dataset loader expects the dataset in Hugging Face format
# If you have a local copy, ensure it follows the standard format with:
# - "question" field: The math problem
# - "answer" field: The solution (with #### prefix for final answer)
```

2. Dataset format:
```
Question: [Math problem text]
Answer: [Solution steps] #### [final numerical answer]
```

Example:
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer: Natalia sold 48 clips in April. In May, she sold half as many clips, which is 48 / 2 = 24 clips. The total number of clips sold is 48 + 24 = 72. #### 72
```

### Preparing OpenThoughts Dataset

The OpenThoughts dataset is available on Hugging Face:
```bash
# The dataset will be automatically downloaded when specified
# Dataset: open-thoughts/OpenThoughts3-1.2M
```

## Model Setup

### Download Models

You need access to two models:
1. **Student Model** - The model to be trained (e.g., `Qwen/Qwen3-8B-Base`)
2. **Teacher Model** - The reference model for guidance (e.g., `Qwen/Qwen3-8B`)

## Running Training

```bash
STUDENT_PATH="Qwen/Qwen3-8B-Base" TEACHER_PATH="Qwen/Qwen3-8B" DATA_PATH="gsm8k" N_GPUS=4 bash scripts/run_opd.sh
```

Or you can specific more parameters: 

```bash
# Using a custom local dataset
torchrun --nproc_per_node=4 -m opd.src.training.run_opd \
  --student_model_path "your-model-path" \
  --teacher_model_path "your-teacher-path" \
  --dataset_name "gsm8k" \
  --data_path "./path/to/your/dataset" \
  --output_dir "./checkpoints/custom_model" \
  --num_epochs 10 \
  --learning_rate 1e-5
```