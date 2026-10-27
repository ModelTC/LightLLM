from typing import Any, ContextManager, Optional, Tuple

import torch
from lightllm.platform.base.runtime import DeviceLike, HardwareBackendRuntime


class CudaLikeRuntime(HardwareBackendRuntime):
    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def dist_backend(self) -> str:
        return "nccl"

    def mem_get_info(self, device: DeviceLike) -> Tuple[int, int]:
        return torch.cuda.mem_get_info(self._parse(device))

    def get_device_properties(self, device: DeviceLike) -> Any:
        return torch.cuda.get_device_properties(self._parse(device))

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def get_device_name(self, device_id: Optional[int] = None) -> str:
        if device_id is None:
            device_id = self.current_device()
        return torch.cuda.get_device_name(device_id)

    def set_device(self, device: DeviceLike) -> None:
        torch.cuda.set_device(self._parse(device))

    def create_stream(self, **kwargs) -> Any:
        return torch.cuda.Stream(**kwargs)

    def stream(self, stream: Any) -> ContextManager[Any]:
        return torch.cuda.stream(stream)

    def current_stream(self, device_id: Optional[int] = None) -> Any:
        if device_id is None:
            device_id = self.current_device()
        return torch.cuda.current_stream(device_id)

    def create_event(self, **kwargs) -> torch.Event:
        return torch.cuda.Event(**kwargs)

    def synchronize(self, device: Optional[DeviceLike] = None) -> None:
        if device is None:
            torch.cuda.synchronize()
            return
        torch.cuda.synchronize(self._parse(device))

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)
