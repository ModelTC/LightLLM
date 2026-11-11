import torch

from .api import DeviceRuntime, Platform


class CudaRuntime(DeviceRuntime):
    platform = Platform.CUDA
    graph_type = "CUDAGraph"
    distributed_backend = "nccl"

    def __init__(self, device):
        super().__init__(device, torch.cuda)

    def capture_graph(self, graph, *, pool=None, memory_saver=None):
        if memory_saver is not None:
            return memory_saver.cuda_graph(graph, pool=pool)
        return self.api.graph(graph, pool=pool)
