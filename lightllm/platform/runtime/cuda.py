import torch
from typing import Any, ContextManager, Optional, Union
from lightllm.platform.base.runtime import BackendRuntime


class CudaRuntime(BackendRuntime):

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def dist_backend(self) -> str:
        return "nccl"

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def get_device_name(self, device_id: Optional[int] = None) -> str:
        device_id = device_id if device_id is not None else self.current_device()
        return torch.cuda.get_device_name(device_id)

    def set_device(self, device: Union[int, str, torch.device]) -> None:
        torch.cuda.set_device(self._parse(device))

    def create_stream(self, **kwargs) -> Any:
        return torch.cuda.Stream(**kwargs)

    def stream(self, stream: Optional[Any] = None) -> ContextManager:
        return torch.cuda.stream(stream)

    def current_stream(self, device_id: Optional[int] = None) -> Any:
        device_id = device_id if device_id is not None else self.current_device()
        return torch.cuda.current_stream(device_id)

    def create_event(self, **kwargs) -> torch.Event:
        return torch.cuda.Event(**kwargs)

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
