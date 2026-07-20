"""Uncompressed asynchronous AllReduce gradient communication."""

import torch
from deepspeed import comm as dist


class AllReduceCommunicator:
    """Start layer-wise AllReduce in autograd hooks and average before step."""

    def __init__(self, model):
        self.model = model.module if hasattr(model, "module") else model
        self.world_size = dist.get_world_size()
        self.param_dict = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.pending = {}
        self.hook_handles = []

    def async_allreduce(self, grad, param_name):
        if param_name in self.pending:
            raise RuntimeError(
                f"parameter {param_name!r} produced more than one gradient in a step"
            )
        reduced = grad.detach()
        if not reduced.is_contiguous():
            reduced = reduced.contiguous()
        work = dist.all_reduce(reduced, async_op=True)
        self.pending[param_name] = (work, reduced)
        return reduced

    def register_hooks(self):
        if self.hook_handles:
            raise RuntimeError("AllReduce communication hooks are already registered")
        for name, parameter in self.param_dict.items():
            self.hook_handles.append(
                parameter.register_hook(
                    lambda grad, param_name=name: self.async_allreduce(grad, param_name)
                )
            )

    def synchronize(self):
        """Wait for AllReduce and install averaged gradients before ``step``."""
        try:
            for param_name, (work, reduced) in self.pending.items():
                work.wait()
                reduced.div_(float(self.world_size))
                self.param_dict[param_name].grad = reduced
        finally:
            self.pending.clear()

    def close(self):
        if self.pending:
            self.synchronize()
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
