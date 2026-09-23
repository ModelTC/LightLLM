from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from lightllm.platform.base.backend import HardwareBackend


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    backend_cls: Type[HardwareBackend]


PLATFORMS: dict[str, PlatformSpec] = {}


def register_platform(name: str) -> Callable[[Type[HardwareBackend]], Type[HardwareBackend]]:
    def decorator(backend_cls: Type[HardwareBackend]) -> Type[HardwareBackend]:
        if name in PLATFORMS:
            raise ValueError(f"Platform {name!r} is already registered.")

        backend_cls.platform_name = name
        PLATFORMS[name] = PlatformSpec(
            name=name,
            backend_cls=backend_cls,
        )
        return backend_cls

    return decorator


def get_platform_spec(name: str) -> PlatformSpec:
    spec = PLATFORMS.get(name)
    if spec is None:
        raise RuntimeError(f"Platform {name!r} is not registered, registered: {sorted(PLATFORMS.keys())}.")
    return spec
