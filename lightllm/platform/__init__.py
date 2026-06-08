from typing import Optional
from lightllm.platform.base.registry import Backend, get_platform_spec

_backend: Optional[Backend] = None


def get_backend() -> Backend:
    global _backend

    if _backend is not None:
        return _backend

    from lightllm.platform.plugins import configure_op_plugins
    from lightllm.utils.envs_utils import get_env_start_args

    configure_op_plugins()

    platform_name = get_env_start_args().hardware_platform
    spec = get_platform_spec(platform_name)

    backend_cls = spec.backend_cls
    _backend = backend_cls()

    if not _backend.runtime.is_available():
        raise RuntimeError(f"Backend {platform_name} is not available")

    return _backend


__all__ = ["get_backend"]
