from abc import ABC, abstractmethod
from typing import Type
from lightllm.platform.base.graph import BackendGraph
from lightllm.platform.base.runtime import BackendRuntime

class Backend(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass 

    @property
    @abstractmethod
    def runtime(self) -> BackendRuntime:
        pass

    @property
    @abstractmethod
    def graph(self) -> BackendGraph:
        pass


class BackendRegistry:

    def __init__(self) -> None:
        self._backend_classes: dict[str, type[Backend]] = {}

    def register(self, name: str, backend_class: callable) -> None:
        if name in self._backend_classes:
            raise ValueError(f"Backend already registered: {name}")
        self._backend_classes[name] = backend_class

    def get_backend_class(self, name: str) -> type[Backend]:
        try:
            return self._backend_classes[name]
        except KeyError as exc:
            raise KeyError(f"Backend is not registered: {name}") from exc
    
    def create_backend(self, name: str) -> Backend:
        return self.get_backend_class(name)()

backend_registry = BackendRegistry()

def register_backend(name: str) -> callable:

    def decorator(backend_cls: Type[Backend]) -> Type[Backend]:
        if not name:
            raise ValueError(
                f"register_backend requires a non-empty name."
            )
        backend_registry.register(name, backend_cls)
        return backend_cls

    return decorator