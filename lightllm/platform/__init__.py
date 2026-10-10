from __future__ import annotations

from typing import Optional

import lightllm.platform.backends  # noqa: F401
from lightllm.platform.base.backend import HardwareBackend
from lightllm.platform.base.registry import get_platform_spec
from lightllm.platform.plugin import configure_plugins
from lightllm.utils.envs_utils import get_env_start_args

_backend: Optional[HardwareBackend] = None


def get_hardware_backend() -> HardwareBackend:
    global _backend

    if _backend is not None:
        return _backend

    configure_plugins()

    platform_name = get_env_start_args().hardware_platform
    spec = get_platform_spec(platform_name)

    backend_cls = spec.load_backend_cls()
    backend = backend_cls()

    if not backend.runtime.is_available():
        raise RuntimeError(f"Hardware backend {backend_cls.__name__} is not available.")

    _backend = backend
    return _backend


__all__ = ["get_hardware_backend"]
