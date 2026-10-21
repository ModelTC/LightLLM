from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

import torch

import lightllm.platform.backends  # noqa: F401
from lightllm.platform.base.backend import HardwareBackend
from lightllm.platform.base.registry import get_platform_spec
from lightllm.platform.plugin import configure_plugins
from lightllm.utils.envs_utils import get_env_start_args

_backend: Optional[HardwareBackend] = None


def detect_platform_name() -> str:
    """Platform of the installed torch build, what --hardware_platform auto resolves to.

    Only the build metadata is read, so the launcher process never initializes a device. CUDA, ROCm (the hip build
    exposes the torch.cuda api) and MUSA (torchada maps torch.cuda onto torch_musa) builds run the cuda path, as with
    the old "cuda" default; --hardware_platform musa still selects the musa op variants explicitly.
    """
    if getattr(torch.version, "xpu", None) is not None:
        return "xpu"
    return "cuda"


def resolve_platform_name(platform_name: Optional[str]) -> str:
    """Map "auto" (or no value) to the detected platform, keep an explicit platform as is."""
    if platform_name in (None, "", "auto"):
        return detect_platform_name()
    return platform_name


@lru_cache(maxsize=None)
def get_platform_name() -> str:
    """--hardware_platform from the start args, or the detected platform when it is auto or the start args are
    absent (unit tests, offline tools)."""
    platform_name = None
    if "LIGHTLLM_START_ARGS" in os.environ:
        platform_name = get_env_start_args().get("hardware_platform", None)
    return resolve_platform_name(platform_name)


def _get_backend() -> HardwareBackend:
    global _backend

    if _backend is None:
        configure_plugins()
        spec = get_platform_spec(get_platform_name())
        backend_cls = spec.load_backend_cls()
        _backend = backend_cls()
    return _backend


def get_hardware_backend() -> HardwareBackend:
    return _get_backend()


def check_hardware_backend_available() -> None:
    """Called when a process binds its device, fails early when --hardware_platform does not match the torch build."""
    backend = _get_backend()
    if not backend.runtime.is_available():
        raise RuntimeError(
            f"Hardware backend {type(backend).__name__} is not available, "
            f"check that --hardware_platform matches the installed torch build."
        )


@lru_cache(maxsize=None)
def get_device_type() -> str:
    """Torch device type of the platform ("cuda", "xpu", ...). Does not require a visible device."""
    return _get_backend().runtime.device_type


@lru_cache(maxsize=None)
def get_device_module() -> Any:
    """Torch device module of the platform (torch.cuda, torch.xpu, ...)."""
    return _get_backend().runtime.device_module


def get_dist_backend() -> str:
    """torch.distributed backend of the platform ("nccl", "xccl", ...)."""
    return _get_backend().runtime.dist_backend


__all__ = [
    "detect_platform_name",
    "resolve_platform_name",
    "get_platform_name",
    "get_hardware_backend",
    "check_hardware_backend_available",
    "get_device_type",
    "get_device_module",
    "get_dist_backend",
]
