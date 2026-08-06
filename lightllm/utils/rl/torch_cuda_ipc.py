"""
Patch torch multiprocessing CUDA tensor reductions to make cross-process
CUDA IPC robust when different processes have different CUDA_VISIBLE_DEVICES.

Torch serializes CUDA tensor IPC handles with a device index. That index is
local to each process, so sender cuda:0 and receiver cuda:0 may refer to
different physical GPUs. We replace the serialized device index with the GPU
UUID on send, then map that UUID back to the receiver's local device index
while rebuilding the tensor.

The patch wraps torch's original reducers and only changes the device argument,
so it avoids copying torch's CUDA IPC serialization implementation.

Copied from:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/patch_torch.py
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Union

import torch
from torch.multiprocessing import reductions


def monkey_patch_torch_reductions():
    """Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed"""

    # Currently, NPU does not support UUID. This has been temporarily commented out,
    # with support expected in the fourth quarter.
    # if _is_npu:
    #     return

    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified

    reductions.init_reductions()


# The torch CUDA IPC rebuild signature has kept the device argument at this
# index for years. Keep this constant in one place because both the global
# monkey patch and local bucketed IPC rebuild path need to rewrite it.
CUDA_IPC_REBUILD_DEVICE_ARG_INDEX = 6
_rebuild_device_fallback: ContextVar[Union[int, None]] = ContextVar("rebuild_device_fallback", default=None)


@contextmanager
def cuda_rebuild_device_fallback(device: Union[int, None]):
    token = _rebuild_device_fallback.set(device)
    try:
        yield
    finally:
        _rebuild_device_fallback.reset(token)


def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(output_args, CUDA_IPC_REBUILD_DEVICE_ARG_INDEX, cuda_device_to_uuid)
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    args = _modify_tuple(args, CUDA_IPC_REBUILD_DEVICE_ARG_INDEX, cuda_device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def get_current_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_current_device_module():
    device_name = get_current_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        return torch.cuda


def get_current_device_id() -> int:
    return get_current_device_module().current_device()


def cuda_device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def cuda_device_from_maybe_uuid(device_maybe_uuid: Union[int, str]) -> int:
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        fallback_device = _rebuild_device_fallback.get()
        if fallback_device is not None:
            return fallback_device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def rebuild_cuda_ipc_tensor(handle: tuple[Callable, tuple], device_id: Union[int, None] = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        list_args[CUDA_IPC_REBUILD_DEVICE_ARG_INDEX] = device_id
    return func(*list_args)


def _modify_tuple(t, index: int, modifier: Callable):
    return *t[:index], modifier(t[index]), *t[index + 1 :]
