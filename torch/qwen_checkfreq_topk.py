"""Qwen2.5 full-checkpoint baseline for comparing against LowDiff.

This entry point intentionally does not create a LowDiff communicator and
keeps DeepSpeed's standard gradient synchronization.  It uses the same data
preprocessing and checkpoint payload as ``Qwen.py`` so that checkpoint cost is
the primary experimental difference.
"""

import argparse
import sys
import time
from pathlib import Path

import deepspeed
import torch
import torch.multiprocessing as mp
from deepspeed import comm as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


TORCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TORCH_DIR))
from qwen_lowdiff_topk import (  # noqa: E402
    causal_lm_collate,
    load_training_dataset,
    tokenize_and_pack,
)
from communicator.topk_allgather import TopKAllGatherCommunicator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepSpeed Qwen2.5 training with frequent full checkpoints"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-path")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", "--learning-rate", type=float, default=2e-5, dest="lr")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--overwrite-tokenized-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--compress-ratio", type=float, default=0.01)
    parser.add_argument(
        "--freq",
        type=int,
        default=1,
        help="Save a full checkpoint every N optimizer steps; 1 saves every step",
    )
    parser.add_argument("--save-dir", default="/ssd/ycx/checkfreq")
    parser.add_argument("--resume-from", help="Path to a full checkpoint made by this script")
    return parser.parse_args()


def _to_cpu(data):
    """Clone a nested checkpoint payload to CPU for the persistence worker."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().clone()
    if isinstance(data, dict):
        return {key: _to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_cpu(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu(value) for value in data)
    return data


def snapshot_persist(queue, snapshot_lock, persist_lock):
    """Original checkfreq-style background full-checkpoint persistence."""
    while True:
        item = queue.get()
        if item is None:
            return
        checkpoint, path, step = item

        with snapshot_lock:
            started = time.time()
            checkpoint = _to_cpu(checkpoint)
            print(f"Full snapshot {step} takes {time.time() - started:.3f}s")

        with persist_lock:
            started = time.time()
            torch.save(checkpoint, path)
            print(f"Full persist {step} takes {time.time() - started:.3f}s: {path}")


def full_checkpoint_path(args, global_step):
    safe_model_name = args.model.rstrip("/").replace("/", "--")
    return Path(args.save_dir) / f"{safe_model_name}_full_step{global_step}.pth.tar"


def main():
    args = parse_args()
    if args.seq_length <= 0 or args.batch_size <= 0:
        raise ValueError("--seq-length and --batch-size must be positive")
    if args.freq < 0:
        raise ValueError("--freq must be zero (disabled) or positive")

    deepspeed.init_distributed()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    set_seed(args.seed + rank)

    if rank == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
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

    # Match checkfreq.py: compressed gradients are decompressed before the
    # optimizer update, while checkpoints remain full model/optimizer states.
    engine.enable_backward_allreduce = False
    communicator = TopKAllGatherCommunicator(engine, k=args.compress_ratio)
    communicator.register_hooks()

    snapshot_lock = persist_lock = checkpoint_queue = checkpoint_process = None
    if rank == 0:
        snapshot_lock = mp.Lock()
        persist_lock = mp.Lock()
        checkpoint_queue = mp.Queue()
        checkpoint_process = mp.Process(
            target=snapshot_persist,
            args=(checkpoint_queue, snapshot_lock, persist_lock),
        )
        checkpoint_process.start()

    try:
        global_step = 0
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            engine.train()
            for batch in loader:
                started = time.time()
                batch = {name: value.to(engine.device, non_blocking=True) for name, value in batch.items()}
                loss = engine(**batch).loss
                engine.backward(loss)
                communicator.decompress()

                if rank == 0:
                    with snapshot_lock:
                        engine.step()
                else:
                    engine.step()

                if rank == 0:
                    print(
                        f"[epoch {epoch + 1}/{args.epochs}] step {global_step} "
                        f"loss={loss.item():.4f} time={time.time() - started:.3f}s"
                    )
                    if args.freq > 0 and global_step % args.freq == 0:
                        if persist_lock.acquire(block=False):
                            checkpoint_queue.put((
                                {
                                    "epoch": epoch + 1,
                                    "global_step": global_step,
                                    "model": engine.module.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                },
                                full_checkpoint_path(args, global_step),
                                global_step,
                            ))
                            persist_lock.release()
                        else:
                            print(f"Full checkpoint {global_step} skipped: previous persist is active")
                global_step += 1
    finally:
        if rank == 0 and checkpoint_queue is not None:
            checkpoint_queue.put(None)
            checkpoint_process.join()


if __name__ == "__main__":
    main()
