"""Reference CV training entry point for the uncompressed LowDiff+ path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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

from communicator.lowdiff_plus_allreduce import LowDiffPlusAllReduceCommunicator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSpeed CV training with uncompressed LowDiff+ checkpointing"
    )
    parser.add_argument("--dataset", choices=("imagenet", "cifar100"), default="imagenet")
    parser.add_argument("--dataset-path", default="/hdd/dataset/cv/imagenet_0908/train")
    parser.add_argument("--model", choices=("resnet50", "resnet101", "vgg16", "vgg19"), default="resnet101")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.0125, dest="lr")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--save-dir", default="/data/lowdiff-plus")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="persist the CPU replica every N optimizer steps; 0 disables persistence",
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--snapshot-threads", type=int, default=4)
    parser.add_argument("--max-pending-steps", type=int, default=2)
    return parser.parse_args()

def build_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    # Load dataset
    if args.dataset == 'imagenet':
        return datasets.ImageFolder(
            '/hdd/dataset/cv/imagenet_0908/train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        )

    elif args.dataset == 'cifar100':
        return datasets.CIFAR100(
            '/hdd/dataset/cv/cifar100/train',
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.267, 0.256, 0.276])
            ])
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

def build_model(name: str) -> nn.Module:
    constructors = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vgg16": models.vgg16_bn,
        "vgg19": models.vgg19_bn,
    }
    return constructors[name]()


def load_checkpoint(
    path: str, model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("framework") != "LowDiff+":
        raise ValueError(f"not a LowDiff+ checkpoint: {path}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["global_step"])


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("--batch-size and --epochs must be positive")
    if args.checkpoint_freq < 0:
        raise ValueError("--checkpoint-freq cannot be negative")

    deepspeed.init_distributed()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    torch.manual_seed(args.seed)

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
    raw_optimizer = optim.Adam(
        raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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
    # LowDiff+ owns synchronization so DeepSpeed must not launch a second
    # reduction for the same gradients.
    engine.enable_backward_allreduce = False

    prefix = f"lowdiff_plus_{args.model}_{args.dataset}"
    communicator = LowDiffPlusAllReduceCommunicator(
        engine,
        optimizer,
        save_dir=args.save_dir,
        checkpoint_prefix=prefix,
        num_snapshot_threads=args.snapshot_threads,
        max_pending_steps=args.max_pending_steps,
    )
    communicator.register_hooks()
    criterion = nn.CrossEntropyLoss()

    try:
        for epoch in range(args.epochs):
            engine.train()
            sampler.set_epoch(epoch)
            for images, targets in loader:
                started = time.perf_counter()
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                output = engine(images)
                loss = criterion(output, targets)
                communicator.begin_step(global_step)
                engine.backward(loss)

                should_checkpoint = (
                    args.checkpoint_freq > 0
                    and (global_step + 1) % args.checkpoint_freq == 0
                )
                scheduled_path = communicator.finish_step(
                    checkpoint=should_checkpoint,
                    metadata={
                        "epoch": epoch,
                        "model_name": args.model,
                        "dataset": args.dataset,
                    },
                )
                engine.step()

                if rank == 0:
                    message = (
                        f"[epoch {epoch + 1}/{args.epochs}] step {global_step + 1} "
                        f"loss={loss.item():.4f} time={time.perf_counter() - started:.3f}s"
                    )
                    if scheduled_path is not None:
                        message += f" checkpoint_scheduled={scheduled_path}"
                    print(message, flush=True)
                global_step += 1
    finally:
        # Normal completion waits for durability.  If backward itself failed,
        # avoid masking that exception with close() while a step is still open.
        if not communicator.step_active:
            communicator.close()


if __name__ == "__main__":
    main()
