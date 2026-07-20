import os
import torch
from deepspeed import comm as dist
# import torch.distributed as dist
import concurrent.futures
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import datetime
import time

class LowDiffTopKCommunicator:
    def __init__(self, model, k=0.01, num_threads=None, save_batch_freq=1, profile=False):
        """
        Initialize the Communicator for Top-K gradient compression with async all_gather.

        Args:
            model (nn.Module): The PyTorch model.
            k (float): Compression ratio (top-k percentage of gradient to keep).
            num_threads (int, optional): Number of threads for decompression. 
                                          Defaults to half of CPU cores.
            batch (int): In-memory batching frequency for saving compressed gradients.
        """
        self.k = k
        self.model = model
        self.compression_data = {}  # Store async work handles and gathered results
        self.profile_enabled = profile
        self.profile_active = False
        self.profile_stats = {}
        
        # Get the number of available CPU threads (default to half of total cores, max 32)
        if num_threads is None:
            num_threads = int(os.cpu_count() / 2)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)  # Thread pool
        self.param_dict = dict(self.model.named_parameters())
        
        print(f"Using {num_threads} threads for gradient decompression.")
        
        if dist.get_rank() == 0:
            self.save_batch_freq = save_batch_freq
            self.diff_ckpt = {}
            self.queue = mp.Queue()
            self.save_process = mp.Process(target=diff_ckpt_saver, args=(self.queue,self.save_batch_freq))
            self.save_process.start()
            print("save process start!")

    def start_profile_step(self):
        """Start collecting host-side compression timing for one training step."""
        if self.profile_enabled:
            self.profile_active = True
            self.profile_stats = {
                "topk_launch_s": 0.0,
                "allgather_launch_s": 0.0,
                "restore_wait_s": 0.0,
                "checkpoint_queue_put_s": 0.0,
            }

    def finish_profile_step(self):
        """Return and reset the current step's host-side timing counters."""
        if not self.profile_active:
            return {}
        stats = self.profile_stats.copy()
        self.profile_active = False
        self.profile_stats = {}
        return stats

    def _profile_add(self, name, elapsed):
        if self.profile_active:
            self.profile_stats[name] = self.profile_stats.get(name, 0.0) + elapsed

    def topk_compress(self, tensor):
        """
        Compress the gradient into Top-K format.
        """
        num_elements = tensor.numel()
        k_elements = max(1, int(num_elements * self.k))

        # torch.topk/gather require int64 indices, but each flattened parameter
        # in this communicator must fit in int32 so indices can be stored and
        # communicated at half the previous cost.
        if num_elements - 1 > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"Parameter with {num_elements} elements exceeds the int32 index range"
            )

        started = time.perf_counter()
        values, indices = torch.topk(tensor.view(-1).abs(), k_elements, sorted=False)
        values = tensor.view(-1).gather(0, indices)
        self._profile_add("topk_launch_s", time.perf_counter() - started)

        return indices.to(torch.int32), values
        
    def async_send(self, grad, param_name):
        """
        Hook function for gradient compression.
        """
        world_size = dist.get_world_size()
        indices, values = self.topk_compress(grad)

        gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
        gathered_values = [torch.zeros_like(values) for _ in range(world_size)]
        
        # Perform async all_gather
        started = time.perf_counter()
        work_indices = dist.all_gather(gathered_indices, indices, async_op=True)
        work_values = dist.all_gather(gathered_values, values, async_op=True)
        self._profile_add("allgather_launch_s", time.perf_counter() - started)
        
        # Store work handles and gathered buffers
        self.compression_data[param_name] = {
            "work_indices": work_indices,
            "work_values": work_values,
            "gathered_indices": gathered_indices,
            "gathered_values": gathered_values,
            "grad_shape": grad.shape
        }
        
        if dist.get_rank() == 0:
            self.diff_ckpt[param_name] = {'values': gathered_values, 'indices': gathered_indices, 'shape': grad.shape}
        
        return None  # Do not modify grad immediately
    
    def register_hooks(self):
        """
        Register Top-K compression hooks for model parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.async_send(grad, name))
    
    def decompress_save(self, diff, filename, i):
        """
        Parallel gradient restoration.
        """
        def process_gradient(param, data):
            data["work_indices"].wait()
            data["work_values"].wait()

            restored_grad = torch.zeros(data["grad_shape"], device=data["gathered_values"][0].device).view(-1)

            for indices, values in zip(data["gathered_indices"], data["gathered_values"]):
                # scatter_add_ requires int64 indices.  Keep the temporary
                # conversion here so the gathered/checkpoint representation
                # remains int32.
                restored_grad.scatter_add_(0, indices.long(), values)

            param.grad = restored_grad.view(data["grad_shape"])  # Direct assignment

        # Submit tasks to the thread pool and wait for completion
        restore_started = time.perf_counter()
        futures = [
            self.executor.submit(process_gradient, self.param_dict[name], data)
            for name, data in self.compression_data.items()
        ]
        concurrent.futures.wait(futures)
        self._profile_add("restore_wait_s", time.perf_counter() - restore_started)

        # Clear stored data
        self.compression_data.clear()
        
        # Send the compressed gradients to the save process
        if diff and dist.get_rank() == 0:
            enqueued_at = time.perf_counter()
            self.queue.put((self.diff_ckpt, filename, i, self.profile_active, enqueued_at))
            self._profile_add("checkpoint_queue_put_s", time.perf_counter() - enqueued_at)

    def __del__(self):
        """
        Ensure the thread pool is properly shut down on object destruction.
        """
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
        if hasattr(self, "queue"):
            self.queue.put(None)
            self.save_process.join()

def diff_ckpt_saver(queue,save_batch_freq):
    """
    Background process that saves compressed gradients to disk.
    
    Args:
        queue (mp.Queue): Queue receiving data to be saved.
        save_batch_freq (int): Save frequency in terms of batch steps.
    """
    
    batch_buffer = {}
    print("batching freq = {}".format(save_batch_freq))
    
    while True:
        data = queue.get()
        
        if data is None:
            break
        if len(data) == 3:
            diff, filename, i = data
            profile, enqueued_at = False, None
        else:
            diff, filename, i, profile, enqueued_at = data
        snapshot_started = time.perf_counter()
        data = _to_cpu(data)
        snapshot_seconds = time.perf_counter() - snapshot_started
    
        if save_batch_freq == 1 :
            begin = time.time()
            torch.save(diff, filename)
            end = time.time()
            now = datetime.datetime.now()
            print("Saved {} time: {:.3f}s at {}".format(filename, end - begin, now))
            if profile:
                print(
                    "PROFILE_CKPT step={} queue_wait_s={:.3f} snapshot_cpu_s={:.3f} "
                    "persist_s={:.3f}".format(
                        i, snapshot_started - enqueued_at, snapshot_seconds, end - begin
                    )
                )
        
        else: 
            batch_buffer[i] = diff
            if i % save_batch_freq == save_batch_freq-1:
                begin = time.time()
                torch.save(batch_buffer, filename)
                end = time.time()
                print("Saved {} time: {:.3f}s".format(filename, end - begin))
                if profile:
                    print(
                        "PROFILE_CKPT step={} queue_wait_s={:.3f} snapshot_cpu_s={:.3f} "
                        "persist_s={:.3f}".format(
                            i, snapshot_started - enqueued_at, snapshot_seconds, end - begin
                        )
                    )
                batch_buffer={}

def _to_cpu(data):
    """
    Move tensor to CPU and return
    """
    if hasattr(data, 'cpu'):
        cpu_data = data.cpu().clone()
        return cpu_data
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    else:
        return data
