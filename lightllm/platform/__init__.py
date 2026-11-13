"""Device operations used by attention; CPU/CUDA adapters are built in."""

import os
from functools import lru_cache

import torch

from .api import DeviceCaps, DeviceRuntime, Platform

_runtime_factories = {}


def register_device_runtime(platform, factory):
    """Register an additional adapter at worker startup, before creating any runtime.

    The factory receives the original device argument and must resolve/validate it
    for its platform. Registering an adapter does not imply provider kernel support.
    """
    platform = Platform(platform)
    if platform in (Platform.CPU, Platform.CUDA) or platform in _runtime_factories:
        raise ValueError(f"device runtime already registered for {platform.value}")
    _runtime_factories[platform] = factory


def get_device_runtime(device=None, *, platform=None):
    platform = Platform(platform) if platform is not None else None
    if platform in _runtime_factories:
        return _runtime_factories[platform](device)
    if platform not in (None, Platform.CPU, Platform.CUDA):
        raise ValueError(f"no registered runtime for {platform.value}; built-in adapters cover CUDA and CPU")
    use_worker_default = device is None
    default_type = "cpu" if platform is Platform.CPU else "cuda"
    if isinstance(device, int):
        device = f"{default_type}:{device}"
    device = device if device is not None else default_type
    if isinstance(device, str) and device.split(":")[0] not in ("cpu", "cuda", "meta"):
        raise ValueError(f"attention backend refactor supports CUDA and CPU devices only, got {device}")
    device = torch.device(device)
    platform = (
        Platform(platform)
        if platform is not None
        else {
            "cpu": Platform.CPU,
            "meta": Platform.CPU,
            "cuda": Platform.CUDA,
        }.get(device.type)
    )
    if platform is None:
        raise ValueError(f"no device runtime for {device}")
    if platform is Platform.CPU and device.type not in ("cpu", "meta"):
        raise ValueError(f"CPU runtime cannot operate on {device}")
    if platform is not Platform.CPU and device.type in ("cpu", "meta"):
        raise ValueError(f"{platform.value} runtime cannot operate on {device}")
    # 保留调用方指定的 CUDA device；没有显式 ordinal 时使用 worker 绑定的 ordinal。
    if device.index is None and device.type not in ("cpu", "meta"):
        index = os.environ.get("LIGHTLLM_CURRENT_DEVICE_ID") if use_worker_default else None
        if index is None:
            api = torch.cuda
            index = api.current_device() if api.is_available() else 0
        device = torch.device(device.type, int(index))
    return _runtime(platform, device)


@lru_cache(maxsize=None)
def _runtime(platform, device):
    if platform is Platform.CPU:
        return DeviceRuntime(device)
    if platform is Platform.CUDA:
        from .cuda import CudaRuntime

        return CudaRuntime(device)
    raise ValueError(f"no device runtime for platform {platform.value}")
