import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kv_buffer_to_kv_buffer(
    mem_num,
    src_mem_index,
    dst_mem_index,
    src_index_stride: tl.constexpr,
    dst_index_stride: tl.constexpr,
    kv_buffer,
    kv_buffer_stride_l,
    kv_buffer_stride_s,
    kv_buffer_stride_d,
    kv_buffer_tail_dim,
    BLOCK: tl.constexpr,
):
    layer_index = tl.program_id(0).to(tl.int64)
    start_index = tl.program_id(1).to(tl.int64)
    grid_num = tl.num_programs(1).to(tl.int64)

    kv_buffer_stride_l = tl.cast(kv_buffer_stride_l, dtype=tl.int64)
    kv_buffer_stride_s = tl.cast(kv_buffer_stride_s, dtype=tl.int64)
    kv_buffer_stride_d = tl.cast(kv_buffer_stride_d, dtype=tl.int64)

    for i in range(start_index, mem_num, grid_num):
        src_mem = tl.load(src_mem_index + i * src_index_stride)
        dst_mem = tl.load(dst_mem_index + i * dst_index_stride)
        for j in range(tl.cdiv(kv_buffer_tail_dim, BLOCK)):
            offs = j * BLOCK + tl.arange(0, BLOCK)
            mask = offs < kv_buffer_tail_dim
            kv_buffer_data = tl.load(
                kv_buffer + layer_index * kv_buffer_stride_l + src_mem * kv_buffer_stride_s + offs, mask=mask
            )
            tl.store(
                kv_buffer + layer_index * kv_buffer_stride_l + dst_mem * kv_buffer_stride_s + offs,
                kv_buffer_data,
                mask=mask,
            )
    return


@triton.jit
def _copy_kv_buffer_pair_to_kv_buffer_pair(
    mem_num,
    src_mem_index,
    dst_mem_index,
    src_index_stride: tl.constexpr,
    dst_index_stride: tl.constexpr,
    k_buffer,
    v_buffer,
    k_buffer_stride_l,
    k_buffer_stride_s,
    k_buffer_stride_d,
    v_buffer_stride_l,
    v_buffer_stride_s,
    v_buffer_stride_d,
    k_buffer_tail_dim,
    v_buffer_tail_dim,
    BLOCK: tl.constexpr,
):
    layer_index = tl.program_id(0).to(tl.int64)
    start_index = tl.program_id(1).to(tl.int64)
    grid_num = tl.num_programs(1).to(tl.int64)

    k_buffer_stride_l = tl.cast(k_buffer_stride_l, dtype=tl.int64)
    k_buffer_stride_s = tl.cast(k_buffer_stride_s, dtype=tl.int64)
    k_buffer_stride_d = tl.cast(k_buffer_stride_d, dtype=tl.int64)
    v_buffer_stride_l = tl.cast(v_buffer_stride_l, dtype=tl.int64)
    v_buffer_stride_s = tl.cast(v_buffer_stride_s, dtype=tl.int64)
    v_buffer_stride_d = tl.cast(v_buffer_stride_d, dtype=tl.int64)

    for i in range(start_index, mem_num, grid_num):
        src_mem = tl.load(src_mem_index + i * src_index_stride)
        dst_mem = tl.load(dst_mem_index + i * dst_index_stride)
        for j in range(tl.cdiv(k_buffer_tail_dim, BLOCK)):
            offs = j * BLOCK + tl.arange(0, BLOCK)
            mask = offs < k_buffer_tail_dim
            k_buffer_data = tl.load(
                k_buffer + layer_index * k_buffer_stride_l + src_mem * k_buffer_stride_s + offs, mask=mask
            )
            tl.store(
                k_buffer + layer_index * k_buffer_stride_l + dst_mem * k_buffer_stride_s + offs,
                k_buffer_data,
                mask=mask,
            )
        for j in range(tl.cdiv(v_buffer_tail_dim, BLOCK)):
            offs = j * BLOCK + tl.arange(0, BLOCK)
            mask = offs < v_buffer_tail_dim
            v_buffer_data = tl.load(
                v_buffer + layer_index * v_buffer_stride_l + src_mem * v_buffer_stride_s + offs, mask=mask
            )
            tl.store(
                v_buffer + layer_index * v_buffer_stride_l + dst_mem * v_buffer_stride_s + offs,
                v_buffer_data,
                mask=mask,
            )
    return


def copy_kv_buffer_to_kv_buffer(
    src_mem_index: torch.Tensor,
    dst_mem_index: torch.Tensor,
    kv_buffer: torch.Tensor,
):
    assert len(src_mem_index) == len(dst_mem_index)
    assert src_mem_index.is_cuda and dst_mem_index.is_cuda and kv_buffer.is_cuda
    kv_buffer = kv_buffer.view(kv_buffer.shape[0], kv_buffer.shape[1], -1).view(dtype=torch.uint8)
    BLOCK = 4096
    layer_num = kv_buffer.shape[0]
    grid = (
        layer_num,
        1024,
    )
    _copy_kv_buffer_to_kv_buffer[grid](
        mem_num=len(src_mem_index),
        src_mem_index=src_mem_index,
        dst_mem_index=dst_mem_index,
        src_index_stride=src_mem_index.stride(0),
        dst_index_stride=dst_mem_index.stride(0),
        kv_buffer=kv_buffer,
        kv_buffer_stride_l=kv_buffer.stride(0),
        kv_buffer_stride_s=kv_buffer.stride(1),
        kv_buffer_stride_d=kv_buffer.stride(2),
        kv_buffer_tail_dim=kv_buffer.shape[-1],
        BLOCK=BLOCK,
    )
    return


def copy_kv_buffer_pair_to_kv_buffer_pair(
    src_mem_index: torch.Tensor,
    dst_mem_index: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
):
    assert len(src_mem_index) == len(dst_mem_index)
    assert src_mem_index.is_cuda and dst_mem_index.is_cuda and k_buffer.is_cuda and v_buffer.is_cuda
    assert k_buffer.shape == v_buffer.shape
    k_buffer = k_buffer.view(k_buffer.shape[0], k_buffer.shape[1], -1).view(dtype=torch.uint8)
    v_buffer = v_buffer.view(v_buffer.shape[0], v_buffer.shape[1], -1).view(dtype=torch.uint8)
    assert k_buffer.shape == v_buffer.shape
    BLOCK = 4096
    layer_num = k_buffer.shape[0]
    grid = (
        layer_num,
        1024,
    )
    _copy_kv_buffer_pair_to_kv_buffer_pair[grid](
        mem_num=len(src_mem_index),
        src_mem_index=src_mem_index,
        dst_mem_index=dst_mem_index,
        src_index_stride=src_mem_index.stride(0),
        dst_index_stride=dst_mem_index.stride(0),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_buffer_stride_l=k_buffer.stride(0),
        k_buffer_stride_s=k_buffer.stride(1),
        k_buffer_stride_d=k_buffer.stride(2),
        v_buffer_stride_l=v_buffer.stride(0),
        v_buffer_stride_s=v_buffer.stride(1),
        v_buffer_stride_d=v_buffer.stride(2),
        k_buffer_tail_dim=k_buffer.shape[-1],
        v_buffer_tail_dim=v_buffer.shape[-1],
        BLOCK=BLOCK,
    )
    return
