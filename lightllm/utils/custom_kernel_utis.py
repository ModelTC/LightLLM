import torch
import triton
import triton.language as tl
from typing import List


def custom_cat(tensors):
    """
    直接调用 torch 的 cat操作，会造成多个流同步阻塞，用 custom_cat 进行替换。
    注意返回的 tensor 为 cpu pin_memory 类型, 只会在一些特殊的场景使用。
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError("Input must be a list of tensors")

    assert tensors[0].is_cuda and len(tensors[0].shape) == 1
    sizes = [t.shape[0] for t in tensors]
    dest_size = sum(sizes)
    out_tensor = torch.empty((dest_size,), dtype=tensors[0].dtype, device="cpu", pin_memory=True)

    start_loc = 0
    for t, size in zip(tensors, sizes):
        out_tensor[start_loc : (start_loc + size)].copy_(t, non_blocking=True)
        start_loc += size

    return out_tensor


def torch_cat_3(tensors: List[torch.Tensor], dim=0):
    if not tensors:
        raise ValueError("at least one tensor")

    ref = tensors[0]
    assert ref.ndim == 3
    dim = dim % ref.ndim

    out = torch.empty(
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


@triton.jit
def _tensor_copy_3dim(
    in_ptr,
    in_stride_0,
    in_stride_1,
    in_stride_2,
    out_ptr,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    head_num,
    head_dim,
    total_len,
    BLOCK_N: tl.constexpr,
):
    start_index = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_N)
    for cur_index in range(start_index, total_len, step=grid_num):
        for cur_head in tl.range(head_num, num_stages=3):
            in_tensor = tl.load(
                in_ptr + in_stride_0 * cur_index + in_stride_1 * cur_head + offs_d, mask=offs_d < head_dim, other=0
            )
            tl.store(
                out_ptr + out_stride_0 * cur_index + out_stride_1 * cur_head + offs_d, in_tensor, mask=offs_d < head_dim
            )
    return


@torch.no_grad()
def tensor_copy_3dim(dest_tensor: torch.Tensor, source_tensor: torch.Tensor):
    assert dest_tensor.shape == source_tensor.shape
    assert dest_tensor.ndim == 3
    assert source_tensor.stride(2) == 1 and dest_tensor.stride(2) == 1
    seq_len, head_num, head_dim = source_tensor.shape
    BLOCK_N = triton.next_power_of_2(head_dim)

    if BLOCK_N <= 256:
        num_warps = 1
    elif BLOCK_N <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    if seq_len <= 16 * 1024:
        grid = (seq_len,)
    else:
        grid = (16 * 1024,)

    _tensor_copy_3dim[grid](
        source_tensor,
        *source_tensor.stride(),
        dest_tensor,
        *dest_tensor.stride(),
        head_num,
        head_dim,
        seq_len,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return
