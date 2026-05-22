from typing import Optional
from lightllm.platform.base.registry import Backend, backend_registry

_backend: Optional[Backend] = None


def _ensure_backends_registered() -> None:
    import lightllm.platform.backends  # noqa: F401


def get_backend() -> Backend:
    global _backend

    if _backend is not None:
        return _backend

    from lightllm.utils.envs_utils import get_env_start_args
    # import backends to register
    _ensure_backends_registered()

    # check if the backend is registered
    platform_name = get_env_start_args().hardware_platform
    try:
        backend_cls = backend_registry.get_backend_class(platform_name)
    except KeyError:
        raise RuntimeError(f"Backend is not registered: {platform_name}")
    _backend = backend_cls()

    # check if the backend is available
    if not _backend.runtime.is_available():
        raise RuntimeError(f"Backend {platform_name} is not available")

    return _backend


__all__ = ["get_backend"]
