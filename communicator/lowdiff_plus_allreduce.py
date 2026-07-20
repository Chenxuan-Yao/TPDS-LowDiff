"""Uncompressed, layer-wise gradient reuse for LowDiff+.

The communicator starts an asynchronous AllReduce from each parameter's
autograd hook.  As soon as a reduction completes, rank zero snapshots that
layer's gradient to host memory.  A background CPU worker applies the complete
step to a CPU-resident model/optimizer replica and periodically persists it.
"""

from __future__ import annotations

import copy
import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from deepspeed import comm as dist


def _to_cpu(value: Any) -> Any:
    """Recursively detach and copy tensors to CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _unwrap_optimizer(optimizer: Any) -> torch.optim.Optimizer:
    """Find the torch optimizer inside a possible DeepSpeed wrapper."""
    current = optimizer
    visited: set[int] = set()
    while not isinstance(current, torch.optim.Optimizer):
        if id(current) in visited or not hasattr(current, "optimizer"):
            raise TypeError(
                "LowDiff+ requires a torch.optim.Optimizer or a wrapper that "
                "exposes it through .optimizer"
            )
        visited.add(id(current))
        current = current.optimizer
    return current


class LowDiffPlusAllReduceCommunicator:
    """AllReduce gradients and maintain a checkpointable CPU model replica.

    ``begin_step`` must be called immediately before each backward pass and
    ``finish_step`` immediately after it.  This implementation intentionally
    models one backward pass as one optimizer step; gradient accumulation must
    therefore be disabled in the training engine.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        save_dir: str | os.PathLike[str],
        checkpoint_prefix: str = "lowdiff_plus",
        checkpoint_rank: int = 0,
        num_snapshot_threads: int = 4,
        max_pending_steps: int = 2,
    ) -> None:
        if num_snapshot_threads <= 0:
            raise ValueError("num_snapshot_threads must be positive")
        if max_pending_steps <= 0:
            raise ValueError("max_pending_steps must be positive")

        self.model = model.module if hasattr(model, "module") else model
        self.optimizer = _unwrap_optimizer(optimizer)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.checkpoint_rank = checkpoint_rank
        self.save_dir = Path(save_dir)
        self.checkpoint_prefix = checkpoint_prefix

        self._parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self._pending: dict[str, tuple[Any, torch.Tensor, Future[None]]] = {}
        self._active_step: int | None = None
        self._hook_handles: list[Any] = []
        self._closed = False
        self._snapshot_executor = ThreadPoolExecutor(
            max_workers=num_snapshot_threads,
            thread_name_prefix="lowdiff-plus-snapshot",
        )
        self._snapshot_stream = None
        self._snapshot_stream_lock = threading.Lock()
        first_cuda_parameter = next(
            (parameter for parameter in self._parameters.values() if parameter.is_cuda),
            None,
        )
        if self.rank == self.checkpoint_rank and first_cuda_parameter is not None:
            self._snapshot_stream = torch.cuda.Stream(device=first_cuda_parameter.device)

        if self.rank == self.checkpoint_rank:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self._cpu_model = copy.deepcopy(self.model).cpu()
            self._cpu_parameters = dict(self._cpu_model.named_parameters())
            self._cpu_buffers = dict(self._cpu_model.named_buffers())
            self._cpu_optimizer = self._make_cpu_optimizer()
            self._update_queue: queue.Queue[tuple[Any, ...]] = queue.Queue()
            self._step_slots = threading.Semaphore(max_pending_steps)
            self._step_events: dict[int, threading.Event] = {}
            self._worker_error: BaseException | None = None
            self._worker = threading.Thread(
                target=self._cpu_update_loop,
                name="lowdiff-plus-cpu-updater",
                daemon=True,
            )
            self._worker.start()

    def _make_cpu_optimizer(self) -> torch.optim.Optimizer:
        gpu_parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        cpu_groups = []
        for group in self.optimizer.param_groups:
            cpu_group = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
            cpu_group["params"] = [
                self._cpu_parameters[gpu_parameter_names[id(parameter)]]
                for parameter in group["params"]
            ]
            # CUDA-only optimizer modes cannot execute on the CPU replica.
            if cpu_group.get("fused"):
                cpu_group["fused"] = False
            if cpu_group.get("capturable"):
                cpu_group["capturable"] = False
            cpu_groups.append(cpu_group)

        defaults = copy.deepcopy(self.optimizer.defaults)
        if defaults.get("fused"):
            defaults["fused"] = False
        if defaults.get("capturable"):
            defaults["capturable"] = False
        cpu_optimizer = type(self.optimizer)(cpu_groups, **defaults)
        cpu_optimizer.load_state_dict(_to_cpu(self.optimizer.state_dict()))
        return cpu_optimizer

    def register_hooks(self) -> None:
        """Register one layer-wise asynchronous AllReduce hook per parameter."""
        if self._hook_handles:
            raise RuntimeError("LowDiff+ hooks have already been registered")
        for name, parameter in self._parameters.items():
            self._hook_handles.append(
                parameter.register_hook(
                    lambda gradient, parameter_name=name: self._allreduce_hook(
                        gradient, parameter_name
                    )
                )
            )

    def begin_step(self, step: int) -> None:
        """Open a step before backward starts."""
        self._ensure_open()
        self._raise_worker_error()
        if self._active_step is not None:
            raise RuntimeError(f"step {self._active_step} has not been finished")
        if self._pending:
            raise RuntimeError("pending gradients remain from the previous step")
        if self.rank == self.checkpoint_rank:
            self._step_slots.acquire()
            self._raise_worker_error()
            if step in self._step_events:
                self._step_slots.release()
                raise ValueError(f"LowDiff+ step identifiers must be unique: {step}")
            self._step_events[step] = threading.Event()
        self._active_step = step

    def _allreduce_hook(self, gradient: torch.Tensor, name: str) -> torch.Tensor:
        if self._active_step is None:
            raise RuntimeError("begin_step() must be called before backward()")
        if name in self._pending:
            raise RuntimeError(f"parameter {name!r} produced more than one gradient in a step")

        reduced = gradient.detach()
        if not reduced.is_contiguous():
            reduced = reduced.contiguous()
        work = dist.all_reduce(reduced, async_op=True)
        future = self._snapshot_executor.submit(
            self._wait_and_snapshot, work, reduced, self._active_step, name
        )
        self._pending[name] = (work, reduced, future)
        return reduced

    def _wait_and_snapshot(
        self,
        work: Any,
        reduced: torch.Tensor,
        step: int,
        name: str,
    ) -> None:
        if self.rank != self.checkpoint_rank:
            work.wait()
            return

        if reduced.is_cuda:
            host_gradient = torch.empty_like(reduced, device="cpu", pin_memory=True)
            # Put the collective dependency and D2H copy on a dedicated stream
            # so snapshotting can overlap later layers' backward computation.
            with self._snapshot_stream_lock:
                assert self._snapshot_stream is not None
                with torch.cuda.stream(self._snapshot_stream):
                    work.wait()
                    host_gradient.copy_(reduced, non_blocking=True)
                    copied = torch.cuda.Event()
                    copied.record(self._snapshot_stream)
                # The queue consumer must never observe an in-flight DMA copy.
                copied.synchronize()
        else:
            work.wait()
            host_gradient = reduced.detach().cpu().clone()
        self._update_queue.put(("gradient", step, name, host_gradient))

    def finish_step(
        self,
        checkpoint: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Finish synchronization and enqueue the CPU update for this step.

        The returned path identifies the scheduled checkpoint.  Persistence is
        asynchronous; call ``wait_for_step`` or ``flush`` when durability is
        required before proceeding.
        """
        self._ensure_open()
        if self._active_step is None:
            raise RuntimeError("begin_step() must be called before finish_step()")
        step = self._active_step

        try:
            # Waiting here is required by the GPU optimizer anyway.  Host copies
            # were launched layer-by-layer and overlap the remaining backward.
            for work, _, future in self._pending.values():
                future.result()
                # On CUDA, Work.wait() establishes the dependency on the
                # calling stream.  Repeat it on the training thread so the
                # gradient division and optimizer step cannot race NCCL.
                work.wait()

            scale = float(self.world_size)
            for name, (_, reduced, _) in self._pending.items():
                reduced.div_(scale)
                self._parameters[name].grad = reduced

            checkpoint_path = None
            if self.rank == self.checkpoint_rank:
                buffer_snapshot = {
                    name: buffer.detach().cpu().clone()
                    for name, buffer in self.model.named_buffers()
                }
                if checkpoint:
                    checkpoint_path = self.save_dir / (
                        f"{self.checkpoint_prefix}_step{step + 1}.pth.tar"
                    )
                self._update_queue.put(
                    (
                        "finish",
                        step,
                        buffer_snapshot,
                        checkpoint_path,
                        dict(metadata or {}),
                    )
                )
            return checkpoint_path
        except BaseException:
            if self.rank == self.checkpoint_rank:
                self._step_events.pop(step, None)
                self._step_slots.release()
            raise
        finally:
            self._pending.clear()
            self._active_step = None

    def _cpu_update_loop(self) -> None:
        gradients: dict[int, dict[str, torch.Tensor]] = {}
        try:
            while True:
                item = self._update_queue.get()
                kind = item[0]
                if kind == "stop":
                    return
                if kind == "gradient":
                    _, step, name, gradient = item
                    gradients.setdefault(step, {})[name] = gradient
                    continue

                _, step, buffers, checkpoint_path, metadata = item
                step_gradients = gradients.pop(step, {})
                self._cpu_optimizer.zero_grad(set_to_none=True)
                for name, gradient in step_gradients.items():
                    self._cpu_parameters[name].grad = gradient.div_(float(self.world_size))
                for name, value in buffers.items():
                    self._cpu_buffers[name].copy_(value)
                self._cpu_optimizer.step()

                if checkpoint_path is not None:
                    self._persist_checkpoint(checkpoint_path, step, metadata)
                self._step_events[step].set()
                self._step_slots.release()
        except BaseException as error:
            self._worker_error = error
            for event in self._step_events.values():
                event.set()
            # Unblock a producer even if every allowed slot was occupied.
            self._step_slots.release()

    def _persist_checkpoint(
        self, path: Path, step: int, metadata: dict[str, Any]
    ) -> None:
        payload = {
            "format_version": 1,
            "framework": "LowDiff+",
            "global_step": step + 1,
            "model": self._cpu_model.state_dict(),
            "optimizer": self._cpu_optimizer.state_dict(),
            "metadata": metadata,
        }
        temporary_path = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        print(f"LowDiff+ checkpoint saved: {path}", flush=True)

    def wait_for_step(self, step: int) -> None:
        """Wait until a CPU update (and its optional checkpoint) completes."""
        if self.rank != self.checkpoint_rank:
            return
        event = self._step_events.get(step)
        if event is None:
            raise ValueError(f"unknown LowDiff+ step: {step}")
        event.wait()
        self._raise_worker_error()

    def flush(self) -> None:
        """Wait for all submitted CPU updates and checkpoint writes."""
        if self.rank != self.checkpoint_rank:
            return
        for step in sorted(self._step_events):
            self._step_events[step].wait()
            self._raise_worker_error()

    def close(self) -> None:
        """Drain background work and release hooks and worker threads."""
        if self._closed:
            return
        if self._active_step is not None:
            raise RuntimeError(f"cannot close while step {self._active_step} is active")
        failure = None
        try:
            self.flush()
        except BaseException as error:
            failure = error
        finally:
            if self.rank == self.checkpoint_rank and self._worker.is_alive():
                self._update_queue.put(("stop",))
                self._worker.join()
            self._snapshot_executor.shutdown(wait=True)
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()
            self._closed = True
        if failure is not None:
            raise failure

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LowDiff+ communicator is closed")

    @property
    def step_active(self) -> bool:
        """Whether a backward step is currently open."""
        return self._active_step is not None

    def _raise_worker_error(self) -> None:
        if self.rank == self.checkpoint_rank and self._worker_error is not None:
            raise RuntimeError("LowDiff+ CPU update worker failed") from self._worker_error

    def __enter__(self) -> "LowDiffPlusAllReduceCommunicator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
