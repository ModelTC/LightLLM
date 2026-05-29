import torch
from typing import Any, ContextManager, Optional, Tuple, Union
from lightllm.platform.base.runtime import BackendRuntime


class AscendRuntime(BackendRuntime):
    
    @property
    def device_type(self) -> str:
        return "npu"

    @property
    def dist_backend(self) -> str:
        return "hccl"

    @property
    def dist_init_passes_device_id(self) -> bool:
        return False

    def mem_get_info(self, device: Union[int, torch.device]) -> Tuple[int, int]:
        return torch.npu.mem_get_info(device)

    def get_device_properties(self, device: Union[int, torch.device]) -> Any:
        return torch.npu.get_device_properties(device)

    def device_count(self) -> int:
        return torch.npu.device_count()
    
    def is_available(self) -> bool:
        return torch.npu.is_available()

    def current_device(self) -> int:
        return torch.npu.current_device()

    def get_device_name(self, device_id: Optional[int] = None) -> str:
        device_id = device_id if device_id is not None else self.current_device()
        return torch.npu.get_device_name(device_id)

    def set_device(self, device: Union[int, str, torch.device]) -> None:
        torch.npu.set_device(self._parse(device))

    def create_stream(self, **kwargs) -> Any:
        return torch.npu.Stream(**kwargs)

    def stream(self, stream: Optional[Any] = None) -> ContextManager:
        return torch.npu.stream(stream)

    def current_stream(self, device_id: Optional[int] = None) -> Any:
        device_id = device_id if device_id is not None else self.current_device()
        return torch.npu.current_stream(device_id)

    def create_event(self, **kwargs) -> torch.Event:
        return torch.npu.Event(**kwargs)

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def empty_cache(self) -> None:
        torch.npu.empty_cache()
