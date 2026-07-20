#!/bin/zsh

# LowDiff+: uncompressed layer-wise AllReduce plus a CPU-resident checkpoint.

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1

# Training parameters
DATASET=imagenet
MODEL=vgg19
EPOCHS=10
BATCH_SIZE=64
LR=0.0125

# Checkpoint parameters
CHECKPOINT_FREQ=50
SAVE_DIR=/ssd/ycx/lowdiff_plus
SNAPSHOT_THREADS=4
MAX_PENDING_STEPS=2
RESUME_FROM=""

# Optional resume argument
RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_ARGS=(--resume-from "$RESUME_FROM")
fi

# Distributed training with DeepSpeed
deepspeed --hostfile=hostfile ./torch/cv_lowdiff_plus_allreduce.py \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --checkpoint-freq "$CHECKPOINT_FREQ" \
    --save-dir "$SAVE_DIR" \
    --snapshot-threads "$SNAPSHOT_THREADS" \
    --max-pending-steps "$MAX_PENDING_STEPS" \
    "${RESUME_ARGS[@]}"
