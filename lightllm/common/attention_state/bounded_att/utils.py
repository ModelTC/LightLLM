"""Bounded attention 页池的存储分配策略。"""

from __future__ import annotations

from math import prod
from weakref import finalize

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

# Portable：本进程内其他 CUDA context 也将该内存识别为 pinned；不代替跨进程注册。
CUDA_HOST_REGISTER_PORTABLE = 1
# Mapped：建立设备可访问映射，供 CUDA/Triton kernel 直接访问主机存储。
CUDA_HOST_REGISTER_MAPPED = 2


# ========================================================================
# 一、页池分配：普通存储或 DP fetch 所需的共享 pinned 存储
# ========================================================================

def create_page_pool_tensor(
    size: tuple[int, ...],
    dtype: torch.dtype,
    device: str | torch.device | int = "cpu",
    pin_memory: bool = True,
) -> torch.Tensor:
    """按 DP fetch 开关创建页池 Tensor，保留调用方指定的设备及 pin_memory 行为。

    CPU 页池在启用 DP fetch 时直接使用共享 Storage，避免普通 pinned Tensor
    经 Queue 导出时重新分配共享存储、复制整块页池，并丢失原地址的 CUDA 注册。
    接收进程仍需为自己的共享映射地址注册 CUDA；注册状态不随 Tensor 传递。
    """
    device = torch.device(device)
    pin_memory = pin_memory and device.type == "cpu"
    shared = device.type == "cpu" and get_env_start_args().enable_dp_prompt_cache_fetch
    nbytes = prod(size) * dtype.itemsize

    # 阶段一：未启用 DP fetch 时沿用 torch 分配；GPU 页池也保持原设备分配方式。
    if not shared:
        tensor = torch.empty(
            size=size,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        reason = "GPU device allocation" if device.type != "cpu" else "DP prompt cache fetch disabled"
        logger.info(
            f"Allocated bounded attention page pool: shape={size}, dtype={dtype}, bytes={nbytes}, "
            f"device={device}, shared=False, pin_memory={pin_memory}, allocator=torch.empty, reason={reason}"
        )
        return tensor

    # 阶段二：直接申请共享字节存储，再绑定无数据区的 Tensor，不经过 share_memory_ 的整块复制。
    # _new_shared 是 PyTorch 私有入口，集中在此使用，便于版本升级时验证或替换。
    storage = torch.UntypedStorage._new_shared(nbytes, device="cpu")
    tensor = torch.empty(size=(0,), dtype=dtype, device="cpu").set_(
        source=storage,
        storage_offset=0,
        size=size,
    )

    # 阶段三：在当前进程原地注册；Portable | Mapped 允许 CUDA 异步搬运及设备映射。
    # 不调用 tensor.pin_memory()，否则会另建私有 pinned 存储，破坏共享关系。
    # 空页池没有数据，无需注册；显式 pin_memory=False 时仅创建共享 CPU 存储。
    if pin_memory:
        register_page_pool_tensor(tensor=tensor)
    logger.info(
        f"Allocated bounded attention page pool: shape={size}, dtype={dtype}, bytes={nbytes}, "
        f"device=cpu, shared=True, pin_memory={pin_memory}, reason=DP prompt cache fetch; "
        "direct shared allocation avoids a full pool copy during IPC export"
    )
    return tensor


# ========================================================================
# 二、本进程注册：首次创建和跨进程接收复用同一入口
# ========================================================================

def register_page_pool_tensor(tensor: torch.Tensor) -> None:
    """将共享 CPU 存储注册到当前进程，已 pinned、空存储或 GPU Tensor 无需处理。

    接收端必须先选定自己的 CUDA device。按整个 Storage 注册，避免多个视图重复注册
    重叠区间；注册和注销信息只保存在本进程，不依赖发送方的对象属性或默认 GPU。
    """
    if tensor.device.type != "cpu" or tensor.numel() == 0 or tensor.is_pinned():
        return
    assert tensor.is_shared(), "page pool registration requires shared CPU storage"
    storage = tensor.untyped_storage()
    ptr = storage.data_ptr()
    nbytes = storage.nbytes()
    device_id = torch.cuda.current_device()
    flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED
    torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(ptr, nbytes, flags))
    # finalizer 绑定 Storage，所有本地 Tensor 及视图释放后才注销，不能绑在单个 Tensor 上。
    # 正常释放时等待 GPU 完成使用；解释器退出时交给进程清理，避免 CUDA 已关闭后再调用。
    cleanup = finalize(
        storage,
        _unregister_page_pool_storage,
        ptr,
        device_id,
    )
    cleanup.atexit = False
    logger.info(
        f"Registered shared bounded attention page pool: bytes={nbytes}, cuda_device={device_id}, "
        "flags=Portable|Mapped; registration applies to this process's mapping"
    )


# ========================================================================
# 三、私有辅助：最后一个本地存储引用释放时注销 CUDA 注册
# ========================================================================


def _unregister_page_pool_storage(ptr: int, device_id: int) -> None:
    """等待当前 GPU 使用结束，再注销本进程的映射；共享内存本身由 Storage 管理。"""
    torch.cuda.synchronize(device=device_id)
    torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(ptr))
    logger.debug(f"Unregistered shared bounded attention page pool: cuda_device={device_id}, ptr={ptr}")
