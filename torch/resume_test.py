#!/usr/bin/env python3
"""Verify that LowDiff recovery reproduces a reference CV checkpoint.

The base checkpoint must represent the state *after* ``--base-batch`` has
been updated.  This script consequently replays differential checkpoints
``base_batch + 1`` through ``target_batch`` (inclusive).

Example:
    python torch/resume_test.py --save-dir /ssd/ycx/lowdiff \\
        --dataset imagenet --model resnet101 --epoch 0 \\
        --base-batch 0 --target-batch 50
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence

import torch
import torch.optim as optim
import torchvision.models as models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay LowDiff checkpoints and compare with a full checkpoint."
    )
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--model", default="vgg19")
    parser.add_argument("--compressor", default="topk")
    parser.add_argument("--compressor-ratio", default=0.01, type=float)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--epoch",
        default=0,
        type=int,
        help="Epoch component used in the checkpoint filenames.",
    )
    parser.add_argument(
        "--base-batch",
        default=0,
        type=int,
        help="Batch index of the full checkpoint used as recovery base.",
    )
    parser.add_argument(
        "--target-batch",
        required=True,
        type=int,
        help="Batch index of the full checkpoint used as the reference.",
    )
    parser.add_argument("--save-batch-freq", default=1, type=int)
    parser.add_argument("--lr", default=0.0125, type=float)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--rtol", default=0.0, type=float)
    parser.add_argument("--atol", default=0.0, type=float)
    return parser.parse_args()


def model_for_name(name):
    constructors = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vgg16": models.vgg16_bn,
        "vgg19": models.vgg19_bn,
    }
    try:
        return constructors[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported CV model: {name}") from exc


def checkpoint_prefix(args):
    return (
        f"{args.model}_{args.dataset}_{args.compressor}_"
        f"{args.compressor_ratio}_{args.epoch}"
    )


def full_checkpoint_path(args, batch):
    return os.path.join(args.save_dir, f"{checkpoint_prefix(args)}_{batch}_full.pth.tar")


def differential_path(args, batch):
    return os.path.join(
        args.save_dir,
        f"{checkpoint_prefix(args)}-{batch}_batch{args.save_batch_freq}.pth.tar",
    )


def load_checkpoint(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # These files are local artifacts produced by this project.  Explicitly
    # disabling weights_only keeps compatibility with optimizer checkpoints.
    return torch.load(path, map_location=device, weights_only=False)


def decompress_gradient(compressed, device):
    shape = compressed["shape"]
    flat = torch.zeros(shape, device=device).view(-1)
    for indices, values in zip(compressed["indices"], compressed["values"]):
        # New LowDiff checkpoints store indices as int32; scatter_add_ needs
        # int64 indices on the destination device.
        flat.scatter_add_(0, indices.to(device=device, dtype=torch.long), values.to(device))
    return flat.view(shape)


def get_step_differential(args, step, cache, device):
    """Return one step's compressed gradient, for either storage format."""
    if args.save_batch_freq == 1:
        return load_checkpoint(differential_path(args, step), device)

    block_end = (step // args.save_batch_freq + 1) * args.save_batch_freq - 1
    if block_end not in cache:
        cache[block_end] = load_checkpoint(differential_path(args, block_end), device)

    block = cache[block_end]
    if step in block:
        return block[step]
    if str(step) in block:  # Tolerate checkpoints written with string keys.
        return block[str(step)]
    raise KeyError(f"Batch {step} is absent from batched differential {block_end}")


def replay_differentials(model, optimizer, args, device):
    parameters = dict(model.named_parameters())
    cache = {}
    for step in range(args.base_batch + 1, args.target_batch + 1):
        compressed = get_step_differential(args, step, cache, device)
        for parameter in parameters.values():
            parameter.grad = None
        for name, entry in compressed.items():
            # ``Communicator`` is created after ``deepspeed.initialize`` in
            # cv.py, so its differential keys are normally ``module.*``.
            # Full checkpoints are saved from ``model.module.state_dict()``,
            # whose keys do not have that prefix.
            parameter = parameters.get(name)
            if parameter is None and name.startswith("module."):
                parameter = parameters.get(name.removeprefix("module."))
            if parameter is None:
                raise KeyError(f"Differential contains unknown parameter: {name}")
            parameter.grad = decompress_gradient(entry, device)
        optimizer.step()


class Comparison:
    def __init__(self):
        self.compared = 0
        self.failures = []
        self.max_abs_error = 0.0

    def add_tensor(self, name, expected, actual, rtol, atol):
        self.compared += 1
        expected = expected.detach().cpu()
        actual = actual.detach().cpu()
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            self.failures.append(f"{name}: shape/dtype {tuple(expected.shape)}/{expected.dtype} != {tuple(actual.shape)}/{actual.dtype}")
            return
        if expected.is_floating_point() or expected.is_complex():
            error = (expected - actual).abs().max().item() if expected.numel() else 0.0
            self.max_abs_error = max(self.max_abs_error, error)
            equal = torch.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True)
        else:
            error = 0.0
            equal = torch.equal(expected, actual)
        if not equal:
            self.failures.append(f"{name}: max_abs_error={error:.8g}")

    def add_value(self, name, expected, actual, rtol, atol):
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            self.add_tensor(name, expected, actual, rtol, atol)
        elif isinstance(expected, Mapping) and isinstance(actual, Mapping):
            expected_keys, actual_keys = set(expected), set(actual)
            if expected_keys != actual_keys:
                self.failures.append(f"{name}: keys differ")
            for key in expected_keys & actual_keys:
                self.add_value(f"{name}.{key}", expected[key], actual[key], rtol, atol)
        elif (
            isinstance(expected, Sequence)
            and not isinstance(expected, (str, bytes))
            and isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes))
        ):
            if len(expected) != len(actual):
                self.failures.append(f"{name}: sequence lengths differ")
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.add_value(f"{name}[{index}]", left, right, rtol, atol)
        elif expected != actual:
            self.failures.append(f"{name}: {expected!r} != {actual!r}")


def print_result(label, result):
    print(
        f"{label}: compared={result.compared}, mismatches={len(result.failures)}, "
        f"max_abs_error={result.max_abs_error:.8g}"
    )
    for failure in result.failures[:10]:
        print(f"  {failure}")
    if len(result.failures) > 10:
        print(f"  ... and {len(result.failures) - 10} more")


def main():
    args = parse_args()
    if args.target_batch < args.base_batch:
        raise ValueError("--target-batch must be greater than or equal to --base-batch")
    if args.save_batch_freq < 1:
        raise ValueError("--save-batch-freq must be at least 1")
    if args.save_batch_freq > 1 and args.target_batch % args.save_batch_freq != args.save_batch_freq - 1:
        raise ValueError(
            "For batched checkpoints, --target-batch must end a saved batch block "
            f"(index congruent to {args.save_batch_freq - 1})."
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    base_path = full_checkpoint_path(args, args.base_batch)
    reference_path = full_checkpoint_path(args, args.target_batch)
    print(f"device: {device}")
    print(f"base checkpoint: {base_path}")
    print(f"reference checkpoint: {reference_path}")

    model = model_for_name(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    base = load_checkpoint(base_path, device)
    reference = load_checkpoint(reference_path, device)
    model.load_state_dict(base["model"])
    optimizer.load_state_dict(base["optimizer"])

    replay_differentials(model, optimizer, args, device)

    recovered_state = model.state_dict()
    parameter_result = Comparison()
    for name in dict(model.named_parameters()):
        parameter_result.add_tensor(
            name,
            reference["model"][name],
            recovered_state[name],
            args.rtol,
            args.atol,
        )
    print_result("model parameters", parameter_result)

    optimizer_result = Comparison()
    optimizer_result.add_value(
        "optimizer",
        reference["optimizer"],
        optimizer.state_dict(),
        args.rtol,
        args.atol,
    )
    print_result("optimizer state", optimizer_result)

    failed = parameter_result.failures or optimizer_result.failures
    if failed:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
