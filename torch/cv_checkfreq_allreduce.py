"""Full-checkpoint baseline using uncompressed layer-wise AllReduce."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from deepspeed import comm as dist

sys.path.append(str(Path(__file__).resolve().parent.parent))

from communicator.allreduce import AllReduceCommunicator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSpeed CV training with uncompressed AllReduce and CheckFreq"
    )
    parser.add_argument("--dataset", choices=("imagenet", "cifar100"), default="imagenet")
    parser.add_argument("--dataset-path", default="/hdd/dataset/cv/imagenet_0908/train")
    parser.add_argument("--model", choices=("resnet50", "resnet101", "vgg16", "vgg19"), default="resnet101")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.001, dest="lr")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--save-dir", default="/data/checkfreq-uncompressed")
    parser.add_argument(
        "--freq",
        "--checkpoint-freq",
        type=int,
        default=1,
        dest="checkpoint_freq",
        help="save a full checkpoint every N optimizer steps; 0 disables persistence",
    )
    parser.add_argument("--max-pending-checkpoints", type=int, default=1)
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    if args.dataset == "imagenet":
        return datasets.ImageFolder(
            args.dataset_path,
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
    return datasets.CIFAR100(
        args.dataset_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
                ),
            ]
        ),
    )


def build_model(name: str) -> nn.Module:
    constructors = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vgg16": models.vgg16_bn,
        "vgg19": models.vgg19_bn,
    }
    return constructors[name]()


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


class AsyncCheckpointWriter:
    """Persist immutable CPU checkpoint snapshots without blocking GPU training."""

    def __init__(self, max_pending: int) -> None:
        self._queue: queue.Queue[tuple[dict[str, Any], Path] | None] = queue.Queue(
            maxsize=max_pending
        )
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, checkpoint: dict[str, Any], path: Path) -> None:
        # A bounded queue preserves every requested CheckFreq checkpoint while
        # preventing an unbounded accumulation of complete model snapshots.
        while True:
            self._raise_error()
            if not self._worker.is_alive():
                raise RuntimeError("checkpoint writer stopped unexpectedly")
            try:
                self._queue.put((checkpoint, path), timeout=0.1)
                break
            except queue.Full:
                continue
        self._raise_error()

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                checkpoint, path = item
                temporary = path.with_name(f".{path.name}.tmp")
                started = time.perf_counter()
                torch.save(checkpoint, temporary)
                os.replace(temporary, path)
                print(
                    f"CheckFreq checkpoint saved: {path} "
                    f"({time.perf_counter() - started:.3f}s)",
                    flush=True,
                )
        except BaseException as error:
            self._error = error

    def close(self) -> None:
        while self._worker.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self._worker.join()
        self._raise_error()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("checkpoint writer failed") from self._error


def checkpoint_path(args: argparse.Namespace, global_step: int) -> Path:
    return Path(args.save_dir) / (
        f"checkfreq_uncompressed_{args.model}_{args.dataset}_step{global_step}.pth.tar"
    )


def snapshot_checkpoint(
    engine: Any, optimizer: Any, epoch: int, global_step: int
) -> dict[str, Any]:
    """Create an immutable CPU copy after the GPU optimizer has updated."""
    started = time.perf_counter()
    snapshot = {
        "format_version": 1,
        "framework": "checkfreq-uncompressed",
        "epoch": epoch,
        "global_step": global_step,
        "model": _to_cpu(engine.module.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
    }
    print(f"CheckFreq snapshot step {global_step}: {time.perf_counter() - started:.3f}s")
    return snapshot


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("framework") != "checkfreq-uncompressed":
        raise ValueError(f"not an uncompressed CheckFreq checkpoint: {path}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["global_step"])


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.checkpoint_freq < 0:
        raise ValueError("--freq cannot be negative")
    if args.max_pending_checkpoints <= 0:
        raise ValueError("--max-pending-checkpoints must be positive")

    deepspeed.init_distributed()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    torch.manual_seed(args.seed)

    if rank == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dataset = build_dataset(args)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    raw_model = build_model(args.model).cuda()
    raw_optimizer = optim.SGD(
        raw_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(args.resume_from, raw_model, raw_optimizer)

    engine, optimizer, _, _ = deepspeed.initialize(
        model=raw_model,
        optimizer=raw_optimizer,
        model_parameters=raw_model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": 1,
        },
    )
    engine.enable_backward_allreduce = False

    communicator = AllReduceCommunicator(engine)
    communicator.register_hooks()
    writer = AsyncCheckpointWriter(args.max_pending_checkpoints) if rank == 0 else None
    criterion = nn.CrossEntropyLoss()

    try:
        for epoch in range(args.epochs):
            engine.train()
            sampler.set_epoch(epoch)
            for images, targets in loader:
                started = time.perf_counter()
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                loss = criterion(engine(images), targets)
                engine.backward(loss)
                communicator.synchronize()
                engine.step()
                global_step += 1

                if rank == 0:
                    if (
                        args.checkpoint_freq > 0
                        and global_step % args.checkpoint_freq == 0
                    ):
                        assert writer is not None
                        writer.submit(
                            snapshot_checkpoint(engine, optimizer, epoch, global_step),
                            checkpoint_path(args, global_step),
                        )
                    print(
                        f"[epoch {epoch + 1}/{args.epochs}] step {global_step} "
                        f"loss={loss.item():.4f} time={time.perf_counter() - started:.3f}s",
                        flush=True,
                    )
    finally:
        communicator.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
