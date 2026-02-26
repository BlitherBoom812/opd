#!/bin/bash

# OPD Training using Hugging Face Trainer API
# Usage: STUDENT_PATH="path/to/student" TEACHER_PATH="path/to/teacher" DATA_PATH="dataset_name" N_GPUS=4 bash scripts/run_opd_trainer.sh
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Set default values if not provided
export STUDENT_PATH="${STUDENT_PATH:-Qwen/Qwen3-8B-Base}"
export TEACHER_PATH="${TEACHER_PATH:-Qwen/Qwen3-8B}"
export DATASET_NAME="${DATASET_NAME:-gsm8k}"
export N_GPUS="${N_GPUS:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/opd_trainer_model}"

echo "Starting OPD training with Trainer API..."
echo "Student Model: $STUDENT_PATH"
echo "Teacher Model: $TEACHER_PATH"
echo "Dataset: $DATASET_NAME"
echo "Number of GPUs: $N_GPUS"
echo "Output Directory: $OUTPUT_DIR"

# Run training using torchrun
torchrun --nproc_per_node=$N_GPUS \
    -m opd.src.training.run_opd_trainer \
    --student_model_path $STUDENT_PATH \
    --teacher_model_path $TEACHER_PATH \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
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