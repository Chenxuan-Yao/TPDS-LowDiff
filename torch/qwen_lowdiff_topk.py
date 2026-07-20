"""Train Qwen2.5 with DeepSpeed and LowDiff gradient checkpoints.

The default workload is WikiText-103, which makes this entry point directly
comparable with ``torch/gpt_lowdiff_topk.py``.  ``--dataset-path`` may instead point to a
local .txt, .json, or .jsonl file for continued pre-training on a private
corpus.
"""

import argparse
import sys
import time
from pathlib import Path

import deepspeed
import torch
from datasets import load_dataset
from deepspeed import comm as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from communicator.lowdiff_topk import LowDiffTopKCommunicator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepSpeed continued pre-training for Qwen2.5 with LowDiff"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B",
        help="Hugging Face model ID or an already-downloaded local model directory",
    )
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Hugging Face dataset name (ignored when --dataset-path is supplied)",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-103-raw-v1",
        help="Hugging Face dataset configuration",
    )
    parser.add_argument(
        "--dataset-path",
        help="Local .txt/.json/.jsonl training corpus; avoids a Hub dataset download",
    )
    parser.add_argument("--text-column", default="text", help="Column containing raw text")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", "--learning-rate", type=float, default=2e-5, dest="lr")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument(
        "--overwrite-tokenized-cache",
        action="store_true",
        help="Ignore the existing Hugging Face tokenization cache and rebuild it",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--compress-ratio", type=float, default=0.01)
    parser.add_argument("--diff", action="store_true", help="Persist LowDiff gradient checkpoints")
    parser.add_argument("--save-batch-freq", type=int, default=1)
    parser.add_argument("--freq", type=int, default=0, help="Save a full checkpoint every N steps")
    parser.add_argument("--save-dir", default="/ssd/ycx/lowdiff")
    parser.add_argument("--resume-from", help="Path to a full checkpoint made by this script")
    return parser.parse_args()


def load_training_dataset(args):
    """Load a local corpus or a Hub dataset and return its train split."""
    if args.dataset_path:
        suffix = Path(args.dataset_path).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            loader = "json"
        elif suffix in {".txt", ".text"}:
            loader = "text"
        else:
            raise ValueError("--dataset-path must end in .txt, .text, .json, or .jsonl")
        dataset = load_dataset(loader, data_files={"train": args.dataset_path})["train"]
    else:
        dataset = load_dataset(args.dataset, args.dataset_config, split="train")

    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column {args.text_column!r} was not found; available columns: "
            f"{dataset.column_names}"
        )
    return dataset


def tokenize_and_pack(dataset, tokenizer, args):
    """Tokenize documents then pack them into fixed-length causal-LM sequences."""
    def tokenize(examples):
        # Preserve a document boundary before concatenation, without introducing
        # padding tokens into the causal-LM loss.
        texts = [str(text) + tokenizer.eos_token for text in examples[args.text_column]]
        return tokenizer(texts, add_special_tokens=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_tokenized_cache,
        desc="Tokenizing corpus",
    )

    def pack(examples):
        # Concatenating first prevents each short document from becoming a padded batch.
        tokens = sum(examples["input_ids"], [])
        usable_length = len(tokens) - (len(tokens) % args.seq_length)
        tokens = tokens[:usable_length]
        chunks = [tokens[i : i + args.seq_length] for i in range(0, usable_length, args.seq_length)]
        return {"input_ids": chunks, "labels": [chunk.copy() for chunk in chunks]}

    packed = tokenized.map(
        pack,
        batched=True,
        remove_columns=tokenized.column_names,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_tokenized_cache,
        desc=f"Packing {args.seq_length}-token sequences",
    )
    if len(packed) == 0:
        raise ValueError("The corpus contains fewer tokens than --seq-length")
    # Without this, PyTorch's default collator transposes each Python list of
    # token IDs into a list of tensors.  Returning tensor rows makes a batch a
    # single [batch_size, seq_length] tensor that can be moved to the GPU.
    packed.set_format(type="torch", columns=["input_ids", "labels"])
    return packed


def causal_lm_collate(examples):
    """Build dense causal-LM tensors regardless of the Dataset cache format."""
    def as_long_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.long)
        return torch.tensor(value, dtype=torch.long)

    return {
        "input_ids": torch.stack([as_long_tensor(example["input_ids"]) for example in examples]),
        "labels": torch.stack([as_long_tensor(example["labels"]) for example in examples]),
    }


def save_full_checkpoint(engine, optimizer, args, epoch, global_step):
    safe_model_name = args.model.rstrip("/").replace("/", "--")
    path = Path(args.save_dir) / f"{safe_model_name}_full_step{global_step}.pth.tar"
    started = time.time()
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": engine.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"Full checkpoint saved to {path} in {time.time() - started:.3f}s")


def main():
    args = parse_args()
    if args.seq_length <= 0 or args.batch_size <= 0 or args.save_batch_freq <= 0:
        raise ValueError("--seq-length, --batch-size, and --save-batch-freq must be positive")

    deepspeed.init_distributed()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    set_seed(args.seed + rank)

    if rank == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Qwen has an EOS token but no pad token by default.  Packed training does
    # not pad, while setting it keeps the tokenizer usable if batching changes.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = tokenize_and_pack(load_training_dataset(args), tokenizer, args)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=causal_lm_collate,
    )

    print(f"[Rank {rank}/{world_size}] Loading {args.model}; {len(dataset)} packed sequences")
    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "weight_decay": args.weight_decay},
        },
    }
    engine, optimizer, _, _ = deepspeed.initialize(
        model=base_model, model_parameters=base_model.parameters(), config=ds_config
    )

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        engine.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if rank == 0:
            print(f"Resumed full checkpoint: {args.resume_from}")

    # LowDiff replaces DeepSpeed's standard gradient synchronization with the
    # compressed all-gather/decompression path below.
    engine.enable_backward_allreduce = False
    communicator = LowDiffTopKCommunicator(
        engine,
        k=args.compress_ratio,
        save_batch_freq=args.save_batch_freq,
    )
    communicator.register_hooks()

    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        engine.train()
        for batch in loader:
            started = time.time()
            batch = {name: value.to(engine.device, non_blocking=True) for name, value in batch.items()}
            loss = engine(**batch).loss
            engine.backward(loss)

            diff_path = Path(args.save_dir) / f"qwen_diff_epoch{epoch}_step{global_step}_batch{args.save_batch_freq}.pth.tar"
            communicator.decompress_save(args.diff, str(diff_path), global_step)
            engine.step()

            if rank == 0:
                print(
                    f"[epoch {epoch + 1}/{args.epochs}] step {global_step} "
                    f"loss={loss.item():.4f} time={time.time() - started:.3f}s"
                )
                if args.freq > 0 and global_step % args.freq == 0:
                    save_full_checkpoint(engine, optimizer, args, epoch, global_step)
            global_step += 1


if __name__ == "__main__":
    main()
