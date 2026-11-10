import torch

from .api import DeviceRuntime, Platform


class MusaRuntime(DeviceRuntime):
    platform = Platform.MUSA
    graph_type = "MUSAGraph"
    distributed_backend = "mccl"

    def __init__(self, device):
        import torch_musa  # noqa: F401

        super().__init__(torch.device("musa", torch.device(device).index), torch.musa)
