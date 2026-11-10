import torch

from .api import DeviceRuntime, Platform


class AscendRuntime(DeviceRuntime):
    platform = Platform.ASCEND
    graph_type = "NPUGraph"
    distributed_backend = "hccl"

    def __init__(self, device):
        import torch_npu  # noqa: F401

        super().__init__(torch.device("npu", torch.device(device).index), torch.npu)
