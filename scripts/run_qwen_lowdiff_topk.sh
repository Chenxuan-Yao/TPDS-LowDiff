#!/bin/zsh

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export HF_DATASETS_CACHE=/ssd/ycx/huggingface-cache/datasets

# Training parameters
MODEL=Qwen/Qwen2.5-1.5B

DATASET=wikitext
DATASET_CONFIG=wikitext-103-raw-v1
DATASET_PATH=/hdd/dataset/nlp/transformer/wikitext-103/train.txt

EPOCHS=1
BATCH_SIZE=1
SEQ_LENGTH=1024
GRADIENT_ACCUMULATION_STEPS=1
LR=2e-5

COMPRESS_RATIO=0.01
FREQ=100
SAVE_BATCH_FREQ=1
SAVE_DIR=/ssd/ycx/lowdiff

# Continued pretraining with DeepSpeed
deepspeed ./torch/qwen_lowdiff_topk.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-config "$DATASET_CONFIG" \
  --dataset-path "$DATASET_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seq-length "$SEQ_LENGTH" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --lr "$LR" \
  --compress-ratio "$COMPRESS_RATIO" \
  --diff \
  --freq "$FREQ" \
  --save-batch-freq "$SAVE_BATCH_FREQ" \
  --save-dir "$SAVE_DIR"
