from abc import ABC, abstractmethod
from typing import Any, ContextManager, Optional, Tuple, Union

import torch
import torch.distributed as dist

DeviceLike = Union[int, str, torch.device]


class HardwareBackendRuntime(ABC):
    @property
    @abstractmethod
    def device_type(self) -> str:
        pass

    @property
    @abstractmethod
    def dist_backend(self) -> str:
        pass

    def dist_init_kwargs(self, target_device: torch.device) -> dict[str, Any]:
        return {}

    def init_process_group(
        self,
        *,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device_id: int,
    ) -> None:
        target_device = self.target_device(device_id)
        self.set_device(target_device)

        kwargs: dict[str, Any] = {
            "backend": self.dist_backend,
            "init_method": f"tcp://{host}:{port}",
            "rank": rank,
            "world_size": world_size,
        }
        kwargs.update(self.dist_init_kwargs(target_device))
        dist.init_process_group(**kwargs)

    @abstractmethod
    def mem_get_info(self, device: DeviceLike) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get_device_properties(self, device: DeviceLike) -> Any:
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

    def _parse(self, device: DeviceLike) -> torch.device:
        if isinstance(device, torch.device):
            _device = device
        elif isinstance(device, int):
            _device = torch.device(self.device_type, device)
        elif isinstance(device, str):
            _device = torch.device(device)
        else:
            raise TypeError(f"Invalid device: {device}")

        if _device.type != self.device_type:
            raise ValueError(f"Expected device type {self.device_type!r}, got {_device.type!r} ({_device})")

        if _device.index is None:
            _device = torch.device(self.device_type, self.current_device())

        return _device

    @abstractmethod
    def set_device(self, device: DeviceLike) -> None:
        pass

    @abstractmethod
    def create_stream(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def stream(self, stream: Any) -> ContextManager[Any]:
        pass

    @abstractmethod
    def current_stream(self, device_id: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def create_event(self, **kwargs) -> torch.Event:
        pass

    @abstractmethod
    def synchronize(self, device: Optional[DeviceLike] = None) -> None:
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        pass

    @abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        pass
