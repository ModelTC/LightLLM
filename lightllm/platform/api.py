from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property

import torch

from lightllm.utils.device_utils import Platform


@dataclass(frozen=True)
class DeviceCaps:
    """实际设备的能力快照；torch 的 device.type 不足以区分 CUDA、ROCm 和兼容层。"""

    platform: Platform
    device: torch.device
    available: bool = True
    device_name: str = ""
    capability: tuple[int, int] | None = None
    supports_graph: bool = False


class DeviceRuntime:
    """一个 worker/device 的设备操作。实现库和模型语义不属于 runtime。

    构造不切换当前设备；调用方先完成 worker 的设备绑定。caps 延迟读取，
    注册表和 CPU 契约测试可以在不初始化设备驱动的情况下导入这些接口。
    """

    platform = Platform.CPU
    graph_type = ""
    distributed_backend = "gloo"

    def __init__(self, device, api=None):
        self.device = torch.device(device)
        self.api = api

    @cached_property
    def caps(self):
        if self.api is None:
            return DeviceCaps(self.platform, self.device, available=self.device.type != "meta")
        available = self.api.is_available()
        capability = None
        if available and hasattr(self.api, "get_device_capability"):
            capability = tuple(self.api.get_device_capability(self.device))
        return DeviceCaps(
            self.platform,
            self.device,
            available,
            self.api.get_device_name(self.device) if available else "",
            capability,
            available and hasattr(self.api, self.graph_type),
        )

    def allocate(self, shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def current_stream(self):
        return self.api.current_stream(self.device) if self.api is not None else None

    def create_stream(self):
        return self.api.Stream(device=self.device) if self.api is not None else None

    def use_stream(self, stream):
        return self.api.stream(stream) if self.api is not None else nullcontext()

    def create_event(self, *, enable_timing=False):
        return self.api.Event(enable_timing=enable_timing) if self.api is not None else None

    def synchronize(self):
        if self.api is not None:
            self.api.synchronize(self.device)

    def is_capturing(self):
        return self.api is not None and self.api.is_current_stream_capturing()

    def empty_cache(self):
        if self.api is not None:
            self.api.empty_cache()

    def memory_info(self):
        if self.api is None:
            raise ValueError("host/meta storage requires an explicit capacity or a memory-budget runtime")
        return self.api.mem_get_info(self.device)

    def graph_pool(self):
        return self.api.graph_pool_handle() if self.caps.supports_graph else None

    def create_graph(self):
        if not self.caps.supports_graph:
            raise RuntimeError(f"{self.platform.value} device {self.device} does not support Graph capture")
        return getattr(self.api, self.graph_type)()

    def capture_graph(self, graph, *, pool=None, memory_saver=None):
        if memory_saver is not None and getattr(memory_saver, "enabled", False):
            raise ValueError(f"memory saver is not supported on {self.platform.value}")
        return self.api.graph(graph, pool=pool)
