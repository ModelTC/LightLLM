import torch
from typing import List

from lightllm.backends import get_device_runtime

_device_runtime = get_device_runtime()


def custom_cat(tensors):
    """
    直接调用 torch 的 cat操作，会造成多个流同步阻塞，用 custom_cat 进行替换。
    注意返回的 tensor 为 cpu pin_memory 类型, 只会在一些特殊的场景使用。
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError("Input must be a list of tensors")

    assert _device_runtime.is_device_tensor(tensors[0]) and len(tensors[0].shape) == 1
    sizes = [t.shape[0] for t in tensors]
    dest_size = sum(sizes)
    out_tensor = _device_runtime.empty((dest_size,), dtype=tensors[0].dtype, device="cpu", pin_memory=True)

    start_loc = 0
    for t, size in zip(tensors, sizes):
        out_tensor[start_loc : (start_loc + size)].copy_(t, non_blocking=_device_runtime.supports_pinned_memory)
        start_loc += size
    _device_runtime.current_stream().synchronize()

    return out_tensor


def torch_cat_3(tensors: List[torch.Tensor], dim=0):
    if not tensors:
        raise ValueError("at least one tensor")

    ref = tensors[0]
    assert ref.ndim == 3
    dim = dim % ref.ndim

    out = _device_runtime.empty(
        [sum(t.size(dim) for t in tensors) if i == dim else ref.size(i) for i in range(ref.ndim)],
        dtype=ref.dtype,
        device=ref.device,
    )

    pos = 0
    for t in tensors:
        if (size := t.size(dim)) > 0:
            slices = [slice(None)] * ref.ndim
            slices[dim] = slice(pos, pos + size)
            tensor_copy_3dim(out[tuple(slices)], t)
            # out[tuple(slices)].copy_(t, non_blocking=True)
            pos += size

    return out


def tensor_copy_3dim(dest_tensor: torch.Tensor, source_tensor: torch.Tensor):
    from lightllm.backends import get_operator_backend

    return get_operator_backend().resolve("tensor_copy_3dim")(dest_tensor, source_tensor)


def pad2dim_tensor_to_new_batch(input: torch.Tensor, new_batch_size: int):
    assert input.ndim == 2
    origin_batch_size = input.shape[0]
    hidden = input.shape[1]

    if origin_batch_size == 0:
        return _device_runtime.zeros(
            (new_batch_size, hidden),
            dtype=input.dtype,
            device=input.device,
            requires_grad=False,
        )

    out = _device_runtime.empty((new_batch_size, hidden), dtype=input.dtype, device=input.device, requires_grad=False)
    out[0:origin_batch_size, :] = input
    out[origin_batch_size:, :] = input[0:1, :]
    return out
