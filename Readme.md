# LowDiff: Efficient Frequent Checkpointing via Low-Cost Differential for High-Performance Distributed Training Systems

## Abstract

This paper proposes LowDiff, an efficient frequent checkpointing framework that reuses compressed gradients, serving as differential checkpoints to reduce cost. Furthermore, LowDiff incorporates a batched gradient write optimization to efficiently persist these differentials to storage. It also dynamically tunes both the checkpoint frequency and the batching size to maximize the performance. Experiments on various workloads show that LowDiff can achieve checkpointing frequency up to once per iteration with less than 3.1\% runtime overhead.

## Architecture

![](LowDiff.png "LowDiff")

## Setup

### Software library
- Ubuntu 22.04
- CUDA-12.4
- NCCL-2.23.4 
- OpenMPI-4.0.5 
- Python >= 3.11
- PyTorch-2.6.0 
- Deepspeed-0.16.4

### Python environment
LowDiff uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

Install uv first if it is not available:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install dependencies from the lockfile:
```
uv sync --locked
```

## Quick start
To run CV jobs:
```
uv run zsh ./scripts/run_cv_lowdiff_topk.sh
```
To run NLP jobs:
```
uv run zsh ./scripts/run_gpt_lowdiff_topk.sh
```

To run the uncompressed LowDiff+ path (layer-wise AllReduce, asynchronous CPU
model updates, and periodic full checkpoints):
```
uv run zsh ./scripts/run_cv_lowdiff_plus_allreduce.sh
```
The checkpoint interval defaults to 50 optimizer steps and can be changed with
`CHECKPOINT_FREQ`.  For example, checkpoint every step and resume later with:
```
CHECKPOINT_FREQ=1 SAVE_DIR=/data/lowdiff-plus uv run zsh ./scripts/run_cv_lowdiff_plus_allreduce.sh
RESUME_FROM=/data/lowdiff-plus/lowdiff_plus_resnet101_imagenet_step100.pth.tar \
  uv run zsh ./scripts/run_cv_lowdiff_plus_allreduce.sh
```
LowDiff+ currently expects one backward pass per optimizer step and a regular
PyTorch optimizer whose update is supported on CPU (the example uses Adam).

For the full-checkpoint CheckFreq baseline with uncompressed AllReduce, run:
```
uv run zsh ./scripts/run_cv_checkfreq_allreduce.sh
```

To run Qwen2.5-1.5B with LowDiff:
```
uv run zsh ./scripts/run_qwen_lowdiff_topk.sh
```
The script defaults to the local WikiText-103 training split at
`/hdd/dataset/nlp/transformer/wikitext-103/train.txt`.  For continued
pre-training on another local corpus, pass a plain-text or JSONL file through
`DATASET_PATH`, for example:
```
DATASET_PATH=/data/corpus.jsonl uv run zsh ./scripts/run_qwen_lowdiff_topk.sh
```
JSON/JSONL input must have a `text` field (or pass `--text-column` when
invoking `torch/qwen_lowdiff_topk.py` directly).  For a representative LowDiff benchmark,
use WikiText-103; for a useful Chinese continued-pretraining run, replace it
with a licensed, cleaned Chinese corpus in the local JSONL format.

The Qwen script caches the tokenized Arrow dataset at
`/ssd/ycx/huggingface-cache/datasets` by default, so later runs with the same
model, corpus, and sequence length reuse tokenization.  Set
`OVERWRITE_TOKENIZED_CACHE=1` to rebuild it, or override `HF_DATASETS_CACHE`
to choose another cache location.

For the Qwen checkfreq baseline (Top-K compressed training gradients and one
full model-and-optimizer checkpoint per optimizer step), run:
```
uv run zsh ./scripts/run_qwen_checkfreq_topk.sh
```

## Datasets
- CIFAR-100: [https://www.cs.utoronto.ca/~kriz/cifar.html](https://www.cs.utoronto.ca/~kriz/cifar.html)
- ImageNet: [https://www.image-net.org/](https://www.image-net.org/)
- Wikitex-2/103: [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)
- SQuAD: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

## License
See [LICENSE](https://github.com/YuchongHu/LowDiff/blob/main/LICENSE).
