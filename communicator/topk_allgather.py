"""Top-K all-gather gradient communication used by the original LowDiff paths."""

import concurrent.futures
import os

import torch
from deepspeed import comm as dist


class TopKAllGatherCommunicator:
    """Compress gradients with Top-K and reconstruct them after all-gather."""

    def __init__(self, model, k=0.01, num_threads=None):
        self.k = k
        self.model = model
        self.compression_data = {}
        if num_threads is None:
            num_threads = max(1, int(os.cpu_count() / 2))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
        self.param_dict = dict(self.model.named_parameters())
        print(f"Using {num_threads} threads for gradient decompression.")

    def topk_compress(self, tensor):
        num_elements = tensor.numel()
        k_elements = max(1, int(num_elements * self.k))
        if num_elements - 1 > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"Parameter with {num_elements} elements exceeds the int32 index range"
            )
        values, indices = torch.topk(tensor.view(-1).abs(), k_elements, sorted=False)
        values = tensor.view(-1).gather(0, indices)
        return indices.to(torch.int32), values

    def async_send(self, grad, param_name):
        world_size = dist.get_world_size()
        indices, values = self.topk_compress(grad)
        gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
        gathered_values = [torch.zeros_like(values) for _ in range(world_size)]
        work_indices = dist.all_gather(gathered_indices, indices, async_op=True)
        work_values = dist.all_gather(gathered_values, values, async_op=True)
        self.compression_data[param_name] = {
            "work_indices": work_indices,
            "work_values": work_values,
            "gathered_indices": gathered_indices,
            "gathered_values": gathered_values,
            "grad_shape": grad.shape,
        }
        return None

    def register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.async_send(grad, name))

    def decompress(self):
        def process_gradient(param, data):
            data["work_indices"].wait()
            data["work_values"].wait()
            restored_grad = torch.zeros(
                data["grad_shape"], device=data["gathered_values"][0].device
            ).view(-1)
            for indices, values in zip(data["gathered_indices"], data["gathered_values"]):
                restored_grad.scatter_add_(0, indices.long(), values)
            param.grad = restored_grad.view(data["grad_shape"])

        futures = [
            self.executor.submit(process_gradient, self.param_dict[name], data)
            for name, data in self.compression_data.items()
        ]
        concurrent.futures.wait(futures)

    def close(self):
        self.executor.shutdown(wait=True)

    def __del__(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
