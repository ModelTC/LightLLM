"""CUDA allocation and capability mechanisms; execution order belongs to Runner."""

import torch

from lightllm.core.contracts.device import DeviceCaps


class CudaRuntime:
    distributed_backend = "nccl"

    @staticmethod
    def set_device(index):
        torch.cuda.set_device(index)

    @staticmethod
    def process_group_options():
        return {}

    @staticmethod
    def borrow_tensor(tensor):
        from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor

        return tensor_to_no_ref_tensor(tensor)

    @staticmethod
    def decode_warmup_length(max_length):
        return 2

    @staticmethod
    def graph_pool():
        return torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None

    @staticmethod
    def create_graph():
        return torch.cuda.CUDAGraph()

    @staticmethod
    def capture_graph(graph, *, pool=None, memory_saver=None):
        if memory_saver is not None:
            return memory_saver.cuda_graph(graph, pool=pool)
        return torch.cuda.graph(graph, pool=pool)

    @staticmethod
    def synchronize():
        torch.cuda.synchronize()

    @staticmethod
    def barrier():
        torch.distributed.barrier()

    @staticmethod
    def create_event(*, enable_timing=False):
        return torch.cuda.Event(enable_timing=enable_timing)

    @staticmethod
    def create_stream():
        return torch.cuda.Stream()

    @staticmethod
    def current_stream():
        return torch.cuda.current_stream()

    @staticmethod
    def empty_cache():
        torch.cuda.empty_cache()

    def __init__(self, index=None):
        if index is not None:
            self.set_device(index)
        self.device = torch.device("cuda", torch.cuda.current_device() if index is None else index)
        capability = torch.cuda.get_device_capability(self.device)
        self.caps = DeviceCaps("cuda", self.device.index, capability, "nvidia", f"sm_{capability[0]}{capability[1]}")

    @staticmethod
    def available_memory(device):
        return torch.cuda.mem_get_info(device)

    def memory_info(self):
        return self.available_memory(self.device)

    def move_batch(self, batch):
        return batch.to_device(self.device)

    def workspace_budget(self, mem_fraction):
        torch.cuda.empty_cache()
        free, total = self.memory_info()
        return max(0, int(free - total * (1 - mem_fraction)))

    def allocate_workspace(self, size):
        return torch.empty(size, dtype=torch.uint8, device=self.device)

    def allocate_tensor(self, shape, dtype):
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        return g_cache_manager.alloc_tensor(shape, dtype, device=self.device)

    def use_stream(self, stream=None):
        return torch.cuda.stream(stream or torch.cuda.current_stream(self.device))

    def record_event(self):
        event = self.create_event()
        event.record(torch.cuda.current_stream(self.device))
        return event
