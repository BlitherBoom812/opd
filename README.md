# On-Policy Distillation Training

## Overview

OPD is a distillation approach that trains a student model to improve upon a teacher model by:
- Using a reference model for stability
- Computing advantages based on reverse KL divergence
- Applying PPO-style clipped policy loss
- Supporting distributed training with FSDP (Fully Sharded Data Parallel)

## Prerequisites

- Python 3.10.12
- PyTorch 2.10.0
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
mkdir -p ./data/gsm8k
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

### Option 1: Original Implementation

```bash
NUM_GPUS=4 bash scripts/run_opd.sh <args>
```

### Option 2: Hugging Face Trainer API Implementation (Recommended)

```bash
NUM_GPUS=4 bash scripts/run_opd_trainer.sh <args>
```
