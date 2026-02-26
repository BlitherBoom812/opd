#!/bin/bash

# OPD Training using fsdp
# Usage: NUM_GPUS=4 bash scripts/run_opd.sh <args>
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Set default values if not provided
export NUM_GPUS="${NUM_GPUS:-4}"

torchrun --nproc_per_node=$NUM_GPUS \
    -m opd.src.training.run_opd \
    --student_model_path Qwen/Qwen3-8B-Base \
    --teacher_model_path Qwen/Qwen3-8B \
    --dataset_name gsm8k \
    --output_dir ./outputs/checkpoints/opd_trainer_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    "$@"