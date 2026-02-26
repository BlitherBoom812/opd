#!/bin/bash

# OPD Training using Hugging Face Trainer API
# Usage: NUM_GPUS=4 bash scripts/run_opd_trainer.sh <args>
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Set default values if not provided
export NUM_GPUS="${NUM_GPUS:-4}"

# Run training using torchrun
torchrun --nproc_per_node=$NUM_GPUS \
    -m opd.src.training.run_opd_trainer \
    --student_model_path Qwen/Qwen3-8B-Base \
    --teacher_model_path Qwen/Qwen3-8B \
    --dataset_name gsm8k \
    --output_dir ./checkpoints/opd_trainer_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --weight_decay 0.0 \
    --logging_steps 1 \
    --save_steps 500 \
    --save_total_limit 2 \
    --max_seq_length 512 \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --bf16 \
    --ddp_find_unused_parameters false \
    --ddp_backend nccl \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config config/fsdp_config.json \
    "$@"