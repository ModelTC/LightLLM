import importlib
from typing import Any, ContextManager, Optional, Tuple

import torch

from lightllm.platform.base.runtime import DeviceLike, HardwareBackendRuntime


class AscendRuntime(HardwareBackendRuntime):
    @property
    def device_type(self) -> str:
        return "npu"

    @property
    def dist_backend(self) -> str:
        return "hccl"

    def mem_get_info(self, device: DeviceLike) -> Tuple[int, int]:
        return torch.npu.mem_get_info(self._parse(device))

    def get_device_properties(self, device: DeviceLike) -> Any:
        return torch.npu.get_device_properties(self._parse(device))

    def device_count(self) -> int:
        return torch.npu.device_count()

    def is_available(self) -> bool:
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            return False
        return torch.npu.is_available()

    def current_device(self) -> int:
        return torch.npu.current_device()

    def get_device_name(self, device_id: Optional[int] = None) -> str:
        if device_id is None:
            device_id = self.current_device()
        return torch.npu.get_device_name(device_id)

    def set_device(self, device: DeviceLike) -> None:
        torch.npu.set_device(self._parse(device))

    def create_stream(self, **kwargs) -> Any:
        return torch.npu.Stream(**kwargs)

    def stream(self, stream: Any) -> ContextManager[Any]:
        return torch.npu.stream(stream)

    def current_stream(self, device_id: Optional[int] = None) -> Any:
        if device_id is None:
            device_id = self.current_device()
        return torch.npu.current_stream(device_id)

    def create_event(self, **kwargs) -> Any:
        return torch.npu.Event(**kwargs)

    def synchronize(self, device: Optional[DeviceLike] = None) -> None:
        if device is None:
            torch.npu.synchronize()
            return
        torch.npu.synchronize(self._parse(device))

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    def manual_seed_all(self, seed: int) -> None:
        torch.npu.manual_seed_all(seed)
