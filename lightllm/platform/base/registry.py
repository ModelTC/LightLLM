from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Type

from lightllm.platform.base.backend import HardwareBackend


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    backend: str

    def load_backend_cls(self) -> Type[HardwareBackend]:
        module_name, class_name = self.backend.rsplit(":", 1)
        module = importlib.import_module(module_name)
        backend_cls = getattr(module, class_name)
        if not issubclass(backend_cls, HardwareBackend):
            raise TypeError(f"Platform {self.name!r} backend {self.backend!r} must be a HardwareBackend subclass.")

        platform_name = getattr(backend_cls, "platform_name", None)
        if platform_name != self.name:
            raise ValueError(
                f"Registered platform {self.name!r} does not match "
                f"{backend_cls.__name__}.platform_name {platform_name!r}."
            )

        return backend_cls


PLATFORMS: dict[str, PlatformSpec] = {}


def register_platform(name: str, backend: str) -> None:
    if name in PLATFORMS:
        raise ValueError(f"Platform {name!r} is already registered.")

    PLATFORMS[name] = PlatformSpec(name=name, backend=backend)


def get_platform_spec(name: str) -> PlatformSpec:
    spec = PLATFORMS.get(name)
    if spec is None:
        raise RuntimeError(f"Platform {name!r} is not registered, registered: {sorted(PLATFORMS.keys())}.")
    return spec
