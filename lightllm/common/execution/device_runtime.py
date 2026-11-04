"""CUDA allocation and capability mechanisms; execution order belongs to Runner."""

import torch

from .specs import DeviceCaps


class CudaRuntime:
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
    def empty_cache():
        torch.cuda.empty_cache()

    def __init__(self):
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.caps = DeviceCaps("cuda", self.device.index, torch.cuda.get_device_capability(self.device))

    def workspace_budget(self, mem_fraction):
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
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
