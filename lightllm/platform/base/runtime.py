import torch
from abc import ABC, abstractmethod
from typing import Any, Optional, ContextManager, Union


class BackendRuntime(ABC):
    
    @property
    @abstractmethod
    def device_type(self) -> str:
        pass

    def target_device(self, device_id: Optional[int] = None) -> torch.device:
        if device_id is None:
            device_id = self.current_device()
        return torch.device(self.device_type, device_id)

    @abstractmethod
    def device_count(self) -> int:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def current_device(self) -> int:
        pass

    @abstractmethod
    def get_device_name(self, device_id: Optional[int] = None) -> str:
        pass

    def _parse(self, device: Union[int, str, torch.device]) -> torch.device:
        if isinstance(device, torch.device):
            _device = device
        elif isinstance(device, int):
            _device = torch.device(self.device_type, device)
        elif isinstance(device, str):
            _device = torch.device(device)
        else:
            raise ValueError(f"Invalid device: {device}")

        if _device.type != self.device_type:
            raise ValueError(
                f"Expected device type {self.device_type!r}, got {_device.type!r} ({_device})"
            )
        return _device

    @abstractmethod
    def set_device(self, device: Union[int, str, torch.device]) -> None:
        pass

    @abstractmethod
    def create_stream(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def stream(self, stream: Optional[Any] = None) -> ContextManager:
        pass

    @abstractmethod
    def current_stream(self, device_id: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def create_event(self, **kwargs) -> torch.Event:
        pass

    @abstractmethod
    def synchronize(self) -> None:
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        pass
