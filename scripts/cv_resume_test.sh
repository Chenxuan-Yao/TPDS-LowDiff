#!/bin/zsh

# Verify: full checkpoint after batch 0 + differential batches 1..50
#         == full checkpoint after batch 50.
#
# Run run_cv_lowdiff_topk.sh first with the same values below.  FREQ must be 50 (or a
# divisor that still produces both the batch-0 and batch-50 full checkpoints).

DATASET=${DATASET:-imagenet}
MODEL=${MODEL:-vgg19}
COMPRESSOR=${COMPRESSOR:-topk}
COMPRESSOR_RATIO=${COMPRESSOR_RATIO:-0.01}
SAVE_DIR=${SAVE_DIR:-/ssd/ycx/lowdiff}
EPOCH=${EPOCH:-0}
BASE_BATCH=${BASE_BATCH:-0}
TARGET_BATCH=${TARGET_BATCH:-50}
SAVE_BATCH_FREQ=${SAVE_BATCH_FREQ:-1}

uv run python ./torch/resume_test.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --compressor "$COMPRESSOR" \
  --compressor-ratio "$COMPRESSOR_RATIO" \
  --save-dir "$SAVE_DIR" \
  --epoch "$EPOCH" \
  --base-batch "$BASE_BATCH" \
  --target-batch "$TARGET_BATCH" \
  --save-batch-freq "$SAVE_BATCH_FREQ" \
  --rtol 0 \
  --atol 0
