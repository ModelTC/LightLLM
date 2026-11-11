"""CUDA/CPU device operations used by the attention backend runtime."""

from functools import lru_cache
import os

import torch

from .api import DeviceCaps, DeviceRuntime, Platform


def get_device_runtime(device=None, *, platform=None):
    platform = Platform(platform) if platform is not None else None
    if platform not in (None, Platform.CPU, Platform.CUDA):
        raise ValueError(f"attention backend refactor supports CUDA and CPU only, got {platform.value}")
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
