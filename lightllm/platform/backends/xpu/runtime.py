from typing import Any, ContextManager, Optional, Tuple

import torch
from lightllm.platform.base.runtime import DeviceLike, HardwareBackendRuntime


class XpuRuntime(HardwareBackendRuntime):
    # FA3 / flashinfer / xformers have no XPU build, the Triton (and torch sdpa for vit) backends run as is.
    supported_att_backends = ("triton", "sdpa")

    @property
    def device_type(self) -> str:
        return "xpu"

    @property
    def dist_backend(self) -> str:
        return "xccl"

    def mem_get_info(self, device: DeviceLike) -> Tuple[int, int]:
        device = self._parse(device)
        free, total = torch.xpu.mem_get_info(device)
        # Level Zero keeps the memory released by empty_cache() in its pool, mem_get_info does not report it as
        # free although this process reuses it. Count it as free, capped by what the torch allocator does not hold.
        reserved = torch.xpu.memory_reserved(device)
        pooled = torch.xpu.max_memory_reserved(device) - reserved
        return min(free + pooled, total - reserved), total

    def get_device_properties(self, device: DeviceLike) -> Any:
        return torch.xpu.get_device_properties(self._parse(device))

    def device_count(self) -> int:
        return torch.xpu.device_count()

    def is_available(self) -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def current_device(self) -> int:
        return torch.xpu.current_device()

    def get_device_name(self, device_id: Optional[int] = None) -> str:
        if device_id is None:
            device_id = self.current_device()
        return torch.xpu.get_device_name(device_id)

    def set_device(self, device: DeviceLike) -> None:
        torch.xpu.set_device(self._parse(device))

    def create_stream(self, **kwargs) -> Any:
        return torch.xpu.Stream(**kwargs)

    def stream(self, stream: Any) -> ContextManager[Any]:
        return torch.xpu.stream(stream)

    def current_stream(self, device_id: Optional[int] = None) -> Any:
        if device_id is None:
            device_id = self.current_device()
        return torch.xpu.current_stream(device_id)

    def create_event(self, **kwargs) -> torch.Event:
        return torch.xpu.Event(**kwargs)

    def synchronize(self, device: Optional[DeviceLike] = None) -> None:
        if device is None:
            torch.xpu.synchronize()
            return
        torch.xpu.synchronize(self._parse(device))

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    def manual_seed_all(self, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)
