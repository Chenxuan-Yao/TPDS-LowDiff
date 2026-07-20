#!/bin/zsh

# CheckFreq baseline with uncompressed layer-wise AllReduce.

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=imagenet
DATASET_PATH=/hdd/dataset/cv/imagenet_0908/train
MODEL=vgg19
EPOCHS=10
BATCH_SIZE=64
LR=0.001

# Checkpoint parameters
CHECKPOINT_FREQ=1
SAVE_DIR=/ssd/ycx/checkfreq-uncompressed
MAX_PENDING_CHECKPOINTS=1
RESUME_FROM=""

# Optional resume argument
RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_ARGS=(--resume-from "$RESUME_FROM")
fi

# Distributed training with DeepSpeed
deepspeed --hostfile=hostfile ./torch/cv_checkfreq_allreduce.py \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --freq "$CHECKPOINT_FREQ" \
    --save-dir "$SAVE_DIR" \
    --max-pending-checkpoints "$MAX_PENDING_CHECKPOINTS" \
    "${RESUME_ARGS[@]}"
