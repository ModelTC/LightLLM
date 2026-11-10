"""设备操作的惰性入口；导入 attention 不会加载其它平台的 SDK。"""

from functools import lru_cache
import importlib

import torch
import os

from .api import DeviceCaps, DeviceRuntime, Platform


def get_device_runtime(device=None, *, platform=None):
    use_worker_default = device is None
    platform = Platform(platform) if platform is not None else None
    default_type = {Platform.CPU: "cpu", Platform.MUSA: "musa", Platform.ASCEND: "npu"}.get(platform, "cuda")
    if isinstance(device, int):
        device = f"{default_type}:{device}"
    device = device if device is not None else default_type
    # 注册 private-use 设备名后才能解析 torch.device("npu"/"musa")。
    if isinstance(device, str) and device.split(":")[0] == "musa":
        importlib.import_module("torch_musa")
    elif isinstance(device, str) and device.split(":")[0] == "npu":
        importlib.import_module("torch_npu")
    device = torch.device(device)
    platform = (
        Platform(platform)
        if platform is not None
        else {
            "cpu": Platform.CPU,
            "meta": Platform.CPU,
            "cuda": Platform.MUSA if getattr(torch.version, "musa", None) else Platform.CUDA,
            "musa": Platform.MUSA,
            "npu": Platform.ASCEND,
        }.get(device.type)
    )
    if platform is None:
        raise ValueError(f"no device runtime for {device}")
    if platform is Platform.CPU and device.type not in ("cpu", "meta"):
        raise ValueError(f"CPU runtime cannot operate on {device}")
    if platform is not Platform.CPU and device.type in ("cpu", "meta"):
        raise ValueError(f"{platform.value} runtime cannot operate on {device}")
    if platform is Platform.CUDA and (device.type == "musa" or getattr(torch.version, "musa", None)):
        raise ValueError("CUDA runtime requested on MUSA; select hardware_platform=musa")
    # 保留调用方指定的设备。已有 MUSA torchada 路径可能使用 cuda 的 tensor 别名。
    if device.index is None and device.type not in ("cpu", "meta"):
        # 显式 "cuda"/"musa"/"npu" 遵循当前设备上下文，不能被 worker 默认值覆盖。
        index = os.environ.get("LIGHTLLM_CURRENT_DEVICE_ID") if use_worker_default else None
        if index is None:
            if platform is Platform.MUSA:
                importlib.import_module("torch_musa")

                api = torch.musa
            elif platform is Platform.ASCEND:
                importlib.import_module("torch_npu")

                api = torch.npu
            else:
                api = torch.cuda
            index = api.current_device() if api.is_available() else 0
        device = torch.device(device.type, int(index))
    return _runtime(platform, device)


@lru_cache(maxsize=None)
def _runtime(platform, device):
    if platform is Platform.CPU:
        return DeviceRuntime(device)
    if platform in (Platform.CUDA, Platform.ROCM):
        from .cuda import CudaRuntime

        return CudaRuntime(device)
    if platform is Platform.MUSA:
        from .musa import MusaRuntime

        return MusaRuntime(device)
    if platform is Platform.ASCEND:
        from .ascend import AscendRuntime

        return AscendRuntime(device)
    raise ValueError(f"no device runtime for platform {platform.value}")
