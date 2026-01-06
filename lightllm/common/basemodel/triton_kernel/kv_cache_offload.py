import torch

import triton
import triton.language as tl
from typing import Optional
import ctypes
from collections import deque


_CUDART = None
_PENDING_D2H_COPIES = deque()


def _get_cudart():
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    # Prefer the default soname; fall back to common CUDA install path.
    try:
        cuda = ctypes.CDLL("libcudart.so")
    except OSError:
        cuda = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so")

    cuda.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    cuda.cudaMemcpyAsync.restype = ctypes.c_int
    cuda.cudaGetErrorString.argtypes = [ctypes.c_int]
    cuda.cudaGetErrorString.restype = ctypes.c_char_p

    _CUDART = cuda
    return _CUDART


def _cuda_memcpy_async(dst_ptr: int, src_ptr: int, nbytes: int, kind: int, stream: int):
    cuda = _get_cudart()
    r = cuda.cudaMemcpyAsync(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(kind),
        ctypes.c_void_p(stream),
    )
    if r != 0:
        err = cuda.cudaGetErrorString(r)
        raise RuntimeError(f"cudaMemcpyAsync failed code={r}, msg={err.decode('utf-8', 'ignore')}")
    return


def _drain_pending_d2h():
    # Keep GPU staging tensors alive until the stream work is complete.
    while _PENDING_D2H_COPIES and _PENDING_D2H_COPIES[0][0].query():
        _PENDING_D2H_COPIES.popleft()
    return


@triton.jit
def _pack_gpu_kv_to_blocks(
    token_indexes_ptr,
    gpu_kv_cache_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    gpu_stride3,
    gpu_kv_cache_scale_ptr,
    gpu_scale_stride0,
    gpu_scale_stride1,
    gpu_scale_stride2,
    gpu_scale_stride3,
    out_kv_cache_ptr,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    out_stride4,
    out_kv_cache_scale_ptr,
    out_scale_stride0,
    out_scale_stride1,
    out_scale_stride2,
    out_scale_stride3,
    out_scale_stride4,
    layer_num,
    head_dim,
    scale_head_dim,
    gpu_token_num,
    block_num,
    cpu_k_start_head_index: tl.constexpr,
    cpu_k_head_num: tl.constexpr,
    gpu_k_start_head_index: tl.constexpr,
    gpu_k_head_num: tl.constexpr,
    cpu_v_start_head_index: tl.constexpr,
    cpu_v_head_num: tl.constexpr,
    gpu_v_start_head_index: tl.constexpr,
    gpu_v_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    block_start_index = tl.program_id(0)
    block_split_size = tl.num_programs(axis=0)

    for block_index in tl.range(block_start_index, block_num, block_split_size):
        token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_indexes = tl.load(token_indexes_ptr + token_range).to(tl.int64)
        token_valid = (token_indexes >= 0) & (token_indexes < gpu_token_num)
        head_dim_range = tl.arange(0, BLOCK_HEAD_DIM)
        head_dim_mask = head_dim_range < head_dim
        scale_head_dim_mask = head_dim_range < scale_head_dim

        for layer_index in range(layer_num):
            for k_head_index in range(gpu_k_head_num):
                gpu_k_head_index = k_head_index + gpu_k_start_head_index
                cpu_k_head_index = k_head_index + cpu_k_start_head_index

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + token_indexes[:, None] * gpu_stride1
                    + gpu_k_head_index.to(tl.int64) * gpu_stride2
                    + head_dim_range[None, :]
                )
                move_mask = token_valid[:, None] & head_dim_mask[None, :]
                gpu_data = tl.load(gpu_ptr, mask=move_mask, other=0.0)
                out_ptr = (
                    out_kv_cache_ptr
                    + block_index * out_stride0
                    + layer_index.to(tl.int64) * out_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * out_stride2
                    + cpu_k_head_index * out_stride3
                    + head_dim_range[None, :]
                )
                tl.store(out_ptr, gpu_data, mask=move_mask, cache_modifier=".wt")
                if HAS_SCALE:
                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + token_indexes[:, None] * gpu_scale_stride1
                        + gpu_k_head_index.to(tl.int64) * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )
                    scale_move_mask = token_valid[:, None] & scale_head_dim_mask[None, :]
                    gpu_scale_data = tl.load(gpu_scale_ptr, mask=scale_move_mask, other=0.0)
                    out_scale_ptr = (
                        out_kv_cache_scale_ptr
                        + block_index * out_scale_stride0
                        + layer_index.to(tl.int64) * out_scale_stride1
                        + tl.arange(0, TOKEN_BLOCK)[:, None] * out_scale_stride2
                        + cpu_k_head_index * out_scale_stride3
                        + head_dim_range[None, :]
                    )
                    tl.store(out_scale_ptr, gpu_scale_data, mask=scale_move_mask, cache_modifier=".wt")

            for v_head_index in range(gpu_v_head_num):
                gpu_v_head_index = v_head_index + gpu_v_start_head_index
                cpu_v_head_index = v_head_index + cpu_v_start_head_index

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + token_indexes[:, None] * gpu_stride1
                    + gpu_v_head_index.to(tl.int64) * gpu_stride2
                    + head_dim_range[None, :]
                )
                move_mask = token_valid[:, None] & head_dim_mask[None, :]
                gpu_data = tl.load(gpu_ptr, mask=move_mask, other=0.0)
                out_ptr = (
                    out_kv_cache_ptr
                    + block_index * out_stride0
                    + layer_index.to(tl.int64) * out_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * out_stride2
                    + cpu_v_head_index * out_stride3
                    + head_dim_range[None, :]
                )
                tl.store(out_ptr, gpu_data, mask=move_mask, cache_modifier=".wt")
                if HAS_SCALE:
                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + token_indexes[:, None] * gpu_scale_stride1
                        + gpu_v_head_index.to(tl.int64) * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )
                    scale_move_mask = token_valid[:, None] & scale_head_dim_mask[None, :]
                    gpu_scale_data = tl.load(gpu_scale_ptr, mask=scale_move_mask, other=0.0)
                    out_scale_ptr = (
                        out_kv_cache_scale_ptr
                        + block_index * out_scale_stride0
                        + layer_index.to(tl.int64) * out_scale_stride1
                        + tl.arange(0, TOKEN_BLOCK)[:, None] * out_scale_stride2
                        + cpu_v_head_index * out_scale_stride3
                        + head_dim_range[None, :]
                    )
                    tl.store(out_scale_ptr, gpu_scale_data, mask=scale_move_mask, cache_modifier=".wt")

    return


@triton.jit
def _pack_gpu_scale_bytes_to_blocks(
    token_indexes_ptr,
    gpu_scale_bytes_ptr,
    gpu_scale_stride0,
    gpu_scale_stride1,
    gpu_scale_stride2,
    gpu_scale_stride3,
    out_bytes_ptr,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    out_stride4,
    layer_num,
    scale_byte_dim,
    gpu_token_num,
    block_num,
    KV_BYTE_OFFSET: tl.constexpr,
    cpu_k_start_head_index: tl.constexpr,
    cpu_k_head_num: tl.constexpr,
    gpu_k_start_head_index: tl.constexpr,
    gpu_k_head_num: tl.constexpr,
    cpu_v_start_head_index: tl.constexpr,
    cpu_v_head_num: tl.constexpr,
    gpu_v_start_head_index: tl.constexpr,
    gpu_v_head_num: tl.constexpr,
    BLOCK_BYTE_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    block_start_index = tl.program_id(0)
    block_split_size = tl.num_programs(axis=0)

    for block_index in tl.range(block_start_index, block_num, block_split_size):
        token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_indexes = tl.load(token_indexes_ptr + token_range).to(tl.int64)
        token_valid = (token_indexes >= 0) & (token_indexes < gpu_token_num)

        byte_range = tl.arange(0, BLOCK_BYTE_DIM)
        byte_mask = byte_range < scale_byte_dim

        for layer_index in range(layer_num):
            for k_head_index in range(gpu_k_head_num):
                gpu_k_head_index = k_head_index + gpu_k_start_head_index
                cpu_k_head_index = k_head_index + cpu_k_start_head_index

                gpu_ptr = (
                    gpu_scale_bytes_ptr
                    + layer_index.to(tl.int64) * gpu_scale_stride0
                    + token_indexes[:, None] * gpu_scale_stride1
                    + gpu_k_head_index.to(tl.int64) * gpu_scale_stride2
                    + byte_range[None, :]
                )
                move_mask = token_valid[:, None] & byte_mask[None, :]
                gpu_bytes = tl.load(gpu_ptr, mask=move_mask, other=0).to(tl.uint8)

                out_ptr = (
                    out_bytes_ptr
                    + block_index * out_stride0
                    + layer_index.to(tl.int64) * out_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * out_stride2
                    + cpu_k_head_index * out_stride3
                    + (KV_BYTE_OFFSET + byte_range)[None, :] * out_stride4
                )
                tl.store(out_ptr, gpu_bytes, mask=move_mask)

            for v_head_index in range(gpu_v_head_num):
                gpu_v_head_index = v_head_index + gpu_v_start_head_index
                cpu_v_head_index = v_head_index + cpu_v_start_head_index

                gpu_ptr = (
                    gpu_scale_bytes_ptr
                    + layer_index.to(tl.int64) * gpu_scale_stride0
                    + token_indexes[:, None] * gpu_scale_stride1
                    + gpu_v_head_index.to(tl.int64) * gpu_scale_stride2
                    + byte_range[None, :]
                )
                move_mask = token_valid[:, None] & byte_mask[None, :]
                gpu_bytes = tl.load(gpu_ptr, mask=move_mask, other=0).to(tl.uint8)

                out_ptr = (
                    out_bytes_ptr
                    + block_index * out_stride0
                    + layer_index.to(tl.int64) * out_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * out_stride2
                    + cpu_v_head_index * out_stride3
                    + (KV_BYTE_OFFSET + byte_range)[None, :] * out_stride4
                )
                tl.store(out_ptr, gpu_bytes, mask=move_mask)

    return


@triton.jit
def _offload_gpu_kv_to_cpu(
    token_indexes_ptr,
    gpu_kv_cache_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    gpu_stride3,
    gpu_kv_cache_scale_ptr,
    gpu_scale_stride0,
    gpu_scale_stride1,
    gpu_scale_stride2,
    gpu_scale_stride3,
    cpu_kv_cache_ptr,
    cpu_stride0,
    cpu_stride1,
    cpu_stride2,
    cpu_stride3,
    cpu_stride4,
    cpu_kv_cache_scale_ptr,
    cpu_scale_stride0,
    cpu_scale_stride1,
    cpu_scale_stride2,
    cpu_scale_stride3,
    cpu_scale_stride4,
    page_indexes_ptr,
    page_readies_ptr,
    layer_num,
    head_dim,
    scale_head_dim,
    block_num,
    cpu_k_start_head_index: tl.constexpr,
    cpu_k_head_num: tl.constexpr,
    gpu_k_start_head_index: tl.constexpr,
    gpu_k_head_num: tl.constexpr,
    cpu_v_start_head_index: tl.constexpr,
    cpu_v_head_num: tl.constexpr,
    gpu_v_start_head_index: tl.constexpr,
    gpu_v_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    block_start_index = tl.program_id(0)
    block_split_size = tl.num_programs(axis=0)

    for block_index in tl.range(block_start_index, block_num, block_split_size):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)

        ready_state = tl.load(page_readies_ptr + block_index)

        mask_layer_num = tl.where(cpu_page_index == -1, 0, 1)
        mask_layer_num = tl.where(ready_state, 0, mask_layer_num)

        token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_indexes = tl.load(token_indexes_ptr + token_range).to(tl.int64)
        head_dim_range = tl.arange(0, BLOCK_HEAD_DIM)
        head_dim_mask = head_dim_range < head_dim
        scale_head_dim_mask = head_dim_range < scale_head_dim

        for layer_index in range(layer_num * mask_layer_num):
            for k_head_index in range(gpu_k_head_num):
                gpu_k_head_index = k_head_index + gpu_k_start_head_index
                cpu_k_head_index = k_head_index + cpu_k_start_head_index

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + token_indexes[:, None] * gpu_stride1
                    + gpu_k_head_index.to(tl.int64) * gpu_stride2
                    + head_dim_range[None, :]
                )
                gpu_data = tl.load(gpu_ptr, mask=head_dim_mask[None, :], other=0.0)
                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_index * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_stride2
                    + cpu_k_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                tl.store(
                    cpu_ptr,
                    gpu_data,
                    mask=head_dim_mask[None, :],
                    cache_modifier=".wt",
                )
                if HAS_SCALE:
                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + token_indexes[:, None] * gpu_scale_stride1
                        + gpu_k_head_index.to(tl.int64) * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )
                    gpu_scale_data = tl.load(gpu_scale_ptr, mask=scale_head_dim_mask[None, :], other=0.0)
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_index * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_scale_stride2
                        + cpu_k_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    tl.store(
                        cpu_scale_ptr,
                        gpu_scale_data,
                        mask=scale_head_dim_mask[None, :],
                        cache_modifier=".wt",
                    )

            for v_head_index in range(gpu_v_head_num):
                gpu_v_head_index = v_head_index + gpu_v_start_head_index
                cpu_v_head_index = v_head_index + cpu_v_start_head_index

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + token_indexes[:, None] * gpu_stride1
                    + gpu_v_head_index.to(tl.int64) * gpu_stride2
                    + head_dim_range[None, :]
                )
                gpu_data = tl.load(gpu_ptr, mask=head_dim_mask[None, :], other=0.0)
                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_index * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_stride2
                    + cpu_v_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                tl.store(
                    cpu_ptr,
                    gpu_data,
                    mask=head_dim_mask[None, :],
                    cache_modifier=".wt",
                )
                if HAS_SCALE:
                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + token_indexes[:, None] * gpu_scale_stride1
                        + gpu_v_head_index.to(tl.int64) * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )
                    gpu_scale_data = tl.load(gpu_scale_ptr, mask=scale_head_dim_mask[None, :], other=0.0)
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_index * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_scale_stride2
                        + cpu_v_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    tl.store(
                        cpu_scale_ptr,
                        gpu_scale_data,
                        mask=scale_head_dim_mask[None, :],
                        cache_modifier=".wt",
                    )

    return


@torch.no_grad()
def offload_gpu_kv_to_cpu(
    token_indexes: torch.Tensor,
    gpu_kv_cache: torch.Tensor,
    gpu_kv_cache_scale: Optional[torch.Tensor],
    cpu_kv_cache: torch.Tensor,
    cpu_kv_cache_scale: Optional[torch.Tensor],
    page_indexes: torch.Tensor,
    page_readies: torch.Tensor,
    tp_index: int,
    tp_world_size: int,
    grid_num: int,
    _cache_data={},
):
    """
    this function is used to offload GPU KV cache to CPU KV cache.
    Args:
        token_indexes: (token_num,)
        gpu_kv_cache: (layer_num, token_num, head_num, head_dim)
        cpu_kv_cache: (all_page_num, layer_num, token_block_size, head_num, head_dim)
        page_indexes: (page_num,)
        page_readies: (page_num,)
    """
    # NOTE: This path intentionally avoids GPU kernels directly dereferencing CPU pointers.
    # We pack KV into a GPU staging tensor, then explicitly D2H-copy into the destination CPU cache pages.
    token_block_size = cpu_kv_cache.shape[2]
    token_num = token_indexes.shape[0]
    assert token_num == page_indexes.shape[0] * token_block_size
    assert page_indexes.shape == page_readies.shape

    gpu_heads = gpu_kv_cache.shape[2]
    gpu_head_dim = gpu_kv_cache.shape[3]
    cpu_heads = cpu_kv_cache.shape[3]
    cpu_head_dim = cpu_kv_cache.shape[4]
    # cpu_head_dim may be a merged head-dim (KV bytes + scale bytes) in int8-kv + fp16-scale cases.
    assert gpu_kv_cache.shape[0] == cpu_kv_cache.shape[1]
    head_dim = gpu_head_dim
    scale_size = (tp_world_size * gpu_heads) // cpu_heads

    # 计算需要拷贝的 head 索引的对应关系
    if (gpu_heads, cpu_heads, tp_index, tp_world_size) in _cache_data:
        need_offload, head_info_tuple = _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)]
    else:
        if cpu_heads > 1:
            assert (tp_world_size * gpu_heads) % cpu_heads == 0
            assert cpu_heads % 2 == 0

            cpu_heads_index = (
                torch.arange(0, cpu_heads, device="cpu", dtype=torch.int32)
                .view(cpu_heads, 1)
                .tile((1, scale_size))
                .view(2, tp_world_size, -1)
            )
            # k
            k_cpu_heads_index = cpu_heads_index[0][tp_index]
            # v
            v_cpu_heads_index = cpu_heads_index[1][tp_index]

            cpu_heads_index = torch.cat([k_cpu_heads_index, v_cpu_heads_index], dim=0).view(2, -1).numpy()
            gpu_heads_index = torch.arange(0, gpu_heads, device="cpu", dtype=torch.int32).view(2, -1).numpy()

            need_offload = tp_index % scale_size == 0

            cpu_k_start_head_index = int(cpu_heads_index[0, 0])
            cpu_k_head_num = len(cpu_heads_index[0])
            gpu_k_start_head_index = int(gpu_heads_index[0, 0])
            gpu_k_head_num = len(gpu_heads_index[0])
            assert cpu_k_head_num == gpu_k_head_num
            cpu_v_start_head_index = int(cpu_heads_index[1, 0])
            cpu_v_head_num = len(cpu_heads_index[1])
            gpu_v_start_head_index = int(gpu_heads_index[1, 0])
            gpu_v_head_num = len(gpu_heads_index[1])
            assert cpu_v_head_num == gpu_v_head_num

            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        else:
            assert gpu_heads == 1
            assert cpu_heads == 1

            need_offload = tp_index == 0

            cpu_k_start_head_index = 0
            cpu_k_head_num = 1
            gpu_k_start_head_index = 0
            gpu_k_head_num = 1
            cpu_v_start_head_index = 0
            cpu_v_head_num = 0
            gpu_v_start_head_index = 0
            gpu_v_head_num = 0
            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)] = (need_offload, head_info_tuple)

    (
        cpu_k_start_head_index,
        cpu_k_head_num,
        gpu_k_start_head_index,
        gpu_k_head_num,
        cpu_v_start_head_index,
        cpu_v_head_num,
        gpu_v_start_head_index,
        gpu_v_head_num,
    ) = head_info_tuple

    if not need_offload:
        return

    assert token_block_size == triton.next_power_of_2(token_block_size)
    block_num = page_indexes.shape[0]

    # Filter out pages that are already ready or invalid.
    # page_indexes/page_readies are expected to be CPU tensors.
    with torch.no_grad():
        block_mask = (~page_readies) & (page_indexes != -1)
        if block_mask.sum().item() == 0:
            return
        block_ids_cpu = torch.nonzero(block_mask, as_tuple=False).view(-1)

    # Gather the corresponding token index blocks on GPU.
    block_ids_gpu = block_ids_cpu.to(device=token_indexes.device, non_blocking=True)
    token_indexes_view = token_indexes.view(block_num, token_block_size)
    token_indexes_sel = token_indexes_view.index_select(0, block_ids_gpu).reshape(-1)
    sel_block_num = int(block_ids_cpu.numel())

    # Allocate GPU staging buffers.
    # If scale exists, prefer packing into a single merged staging tensor so we can do one D2H copy per page.
    HAS_SCALE_TENSOR = gpu_kv_cache_scale is not None and cpu_kv_cache_scale is not None
    HAS_MERGED_SCALE = gpu_kv_cache_scale is not None and (not HAS_SCALE_TENSOR) and (cpu_head_dim > head_dim)

    if HAS_MERGED_SCALE:
        # merged layout uses cpu_kv_cache.dtype as the backing dtype (typically int8/uint8)
        assert cpu_kv_cache.dtype in (
            torch.int8,
            torch.uint8,
        ), "merged KV+scale requires byte-addressable cpu_kv_cache dtype"
        scale_head_dim = gpu_kv_cache_scale.shape[-1]
        scale_bytes_per_head = scale_head_dim * gpu_kv_cache_scale.element_size()
        kv_bytes_per_head = head_dim * cpu_kv_cache.element_size()
        merged_bytes_per_head = cpu_head_dim * cpu_kv_cache.element_size()
        assert (
            kv_bytes_per_head + scale_bytes_per_head == merged_bytes_per_head
        ), "cpu merged head dim bytes must equal kv bytes + scale bytes"

        gpu_out = torch.empty(
            (sel_block_num, cpu_kv_cache.shape[1], token_block_size, cpu_heads, cpu_head_dim),
            device=gpu_kv_cache.device,
            dtype=cpu_kv_cache.dtype,
        )
        out_scale = None
        out_scale_stride = [0 for _ in range(10)]
        gpu_scale_stride = [0 for _ in range(10)]
        scale_byte_dim = int(scale_bytes_per_head)
    else:
        # non-merged path (either no scale, or legacy separate scale tensor)
        if gpu_kv_cache_scale is None:
            assert gpu_head_dim == cpu_head_dim
        gpu_out = torch.empty(
            (sel_block_num, cpu_kv_cache.shape[1], token_block_size, cpu_heads, head_dim),
            device=gpu_kv_cache.device,
            dtype=cpu_kv_cache.dtype,
        )
        if HAS_SCALE_TENSOR:
            scale_head_dim = gpu_kv_cache_scale.shape[-1]
            gpu_scale_stride = gpu_kv_cache_scale.stride()
            out_scale = torch.empty(
                (
                    sel_block_num,
                    cpu_kv_cache_scale.shape[1],
                    token_block_size,
                    cpu_kv_cache_scale.shape[3],
                    cpu_kv_cache_scale.shape[4],
                ),
                device=gpu_kv_cache_scale.device,
                dtype=cpu_kv_cache_scale.dtype,
            )
            out_scale_stride = out_scale.stride()
        else:
            scale_head_dim = 0
            gpu_scale_stride = [0 for _ in range(10)]
            out_scale = None
            out_scale_stride = [0 for _ in range(10)]

    grid = (grid_num,)
    num_warps = 4

    _pack_gpu_kv_to_blocks[grid](
        token_indexes_ptr=token_indexes_sel,
        gpu_kv_cache_ptr=gpu_kv_cache,
        gpu_stride0=gpu_kv_cache.stride(0),
        gpu_stride1=gpu_kv_cache.stride(1),
        gpu_stride2=gpu_kv_cache.stride(2),
        gpu_stride3=gpu_kv_cache.stride(3),
        gpu_kv_cache_scale_ptr=gpu_kv_cache_scale,
        gpu_scale_stride0=gpu_scale_stride[0],
        gpu_scale_stride1=gpu_scale_stride[1],
        gpu_scale_stride2=gpu_scale_stride[2],
        gpu_scale_stride3=gpu_scale_stride[3],
        out_kv_cache_ptr=gpu_out,
        out_stride0=gpu_out.stride(0),
        out_stride1=gpu_out.stride(1),
        out_stride2=gpu_out.stride(2),
        out_stride3=gpu_out.stride(3),
        out_stride4=gpu_out.stride(4),
        out_kv_cache_scale_ptr=out_scale,
        out_scale_stride0=out_scale_stride[0],
        out_scale_stride1=out_scale_stride[1],
        out_scale_stride2=out_scale_stride[2],
        out_scale_stride3=out_scale_stride[3],
        out_scale_stride4=out_scale_stride[4],
        layer_num=gpu_kv_cache.shape[0],
        head_dim=head_dim,
        scale_head_dim=(0 if HAS_MERGED_SCALE else scale_head_dim),
        gpu_token_num=gpu_kv_cache.shape[1],
        block_num=sel_block_num,
        cpu_k_start_head_index=cpu_k_start_head_index,
        cpu_k_head_num=cpu_k_head_num,
        gpu_k_start_head_index=gpu_k_start_head_index,
        gpu_k_head_num=gpu_k_head_num,
        cpu_v_start_head_index=cpu_v_start_head_index,
        cpu_v_head_num=cpu_v_head_num,
        gpu_v_start_head_index=gpu_v_start_head_index,
        gpu_v_head_num=gpu_v_head_num,
        BLOCK_HEAD_DIM=triton.next_power_of_2(head_dim),
        TOKEN_BLOCK=token_block_size,
        HAS_SCALE=(HAS_SCALE_TENSOR and (not HAS_MERGED_SCALE)),
        num_warps=num_warps,
        num_stages=1,
    )

    if HAS_MERGED_SCALE:
        # Pack scale bytes into the merged staging buffer after the KV bytes region.
        # We reinterpret the scale tensor as uint8 so this is a byte-exact copy.
        gpu_scale_bytes = gpu_kv_cache_scale.view(torch.uint8)
        out_bytes = gpu_out.view(torch.uint8)
        _pack_gpu_scale_bytes_to_blocks[grid](
            token_indexes_ptr=token_indexes_sel,
            gpu_scale_bytes_ptr=gpu_scale_bytes,
            gpu_scale_stride0=gpu_scale_bytes.stride(0),
            gpu_scale_stride1=gpu_scale_bytes.stride(1),
            gpu_scale_stride2=gpu_scale_bytes.stride(2),
            gpu_scale_stride3=gpu_scale_bytes.stride(3),
            out_bytes_ptr=out_bytes,
            out_stride0=out_bytes.stride(0),
            out_stride1=out_bytes.stride(1),
            out_stride2=out_bytes.stride(2),
            out_stride3=out_bytes.stride(3),
            out_stride4=out_bytes.stride(4),
            layer_num=gpu_kv_cache.shape[0],
            scale_byte_dim=scale_byte_dim,
            gpu_token_num=gpu_kv_cache.shape[1],
            block_num=sel_block_num,
            KV_BYTE_OFFSET=kv_bytes_per_head,
            cpu_k_start_head_index=cpu_k_start_head_index,
            cpu_k_head_num=cpu_k_head_num,
            gpu_k_start_head_index=gpu_k_start_head_index,
            gpu_k_head_num=gpu_k_head_num,
            cpu_v_start_head_index=cpu_v_start_head_index,
            cpu_v_head_num=cpu_v_head_num,
            gpu_v_start_head_index=gpu_v_start_head_index,
            gpu_v_head_num=gpu_v_head_num,
            BLOCK_BYTE_DIM=triton.next_power_of_2(scale_byte_dim),
            TOKEN_BLOCK=token_block_size,
            num_warps=num_warps,
            num_stages=1,
        )

    # Explicit D2H copies into destination CPU pages (no mapped / zerocopy).
    # IMPORTANT: SHM-backed CPU tensors are not allocated via PyTorch pinned allocator, so
    # tensor.copy_(non_blocking=True) will not be async. We call cudaMemcpyAsync directly.
    _drain_pending_d2h()
    stream = torch.cuda.current_stream().cuda_stream
    CUDA_MEMCPY_DEVICE_TO_HOST = 2
    for i in range(sel_block_num):
        dst_page = int(page_indexes[block_ids_cpu[i]].item())
        if dst_page < 0 or dst_page >= cpu_kv_cache.shape[0]:
            raise RuntimeError(f"cpu_kv_cache page index out of range: {dst_page}, page_num={cpu_kv_cache.shape[0]}")
        dst_view = cpu_kv_cache[dst_page]
        src_view = gpu_out[i]
        assert dst_view.is_contiguous(), "cpu_kv_cache page view must be contiguous"
        assert src_view.is_contiguous(), "gpu staging must be contiguous"
        _cuda_memcpy_async(
            dst_ptr=dst_view.data_ptr(),
            src_ptr=src_view.data_ptr(),
            nbytes=src_view.numel() * src_view.element_size(),
            kind=CUDA_MEMCPY_DEVICE_TO_HOST,
            stream=stream,
        )
        if HAS_SCALE_TENSOR:
            dst_scale_view = cpu_kv_cache_scale[dst_page]
            src_scale_view = out_scale[i]
            assert dst_scale_view.is_contiguous(), "cpu_kv_cache_scale page view must be contiguous"
            assert src_scale_view.is_contiguous(), "gpu scale staging must be contiguous"
            _cuda_memcpy_async(
                dst_ptr=dst_scale_view.data_ptr(),
                src_ptr=src_scale_view.data_ptr(),
                nbytes=src_scale_view.numel() * src_scale_view.element_size(),
                kind=CUDA_MEMCPY_DEVICE_TO_HOST,
                stream=stream,
            )

    # Keep staging buffers alive until the stream reaches this point.
    sync_event = torch.cuda.Event()
    sync_event.record()
    _PENDING_D2H_COPIES.append((sync_event, gpu_out, out_scale))
    return


@triton.jit
def _load_cpu_cache_to_gpu(
    gpu_mem_indexes_ptr,
    copy_token_num,
    copy_block_num,
    cpu_mem_indexes_ptr,
    cpu_page_indexes_ptr,
    gpu_kv_cache_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    gpu_stride3,
    gpu_kv_cache_scale_ptr,
    gpu_scale_stride0,
    gpu_scale_stride1,
    gpu_scale_stride2,
    gpu_scale_stride3,
    cpu_kv_cache_ptr,
    cpu_stride0,
    cpu_stride1,
    cpu_stride2,
    cpu_stride3,
    cpu_stride4,
    cpu_kv_cache_scale_ptr,
    cpu_scale_stride0,
    cpu_scale_stride1,
    cpu_scale_stride2,
    cpu_scale_stride3,
    cpu_scale_stride4,
    layer_num,
    head_dim,
    scale_head_dim,
    cpu_k_start_head_index: tl.constexpr,
    cpu_k_head_num: tl.constexpr,
    gpu_k_start_head_index: tl.constexpr,
    gpu_k_head_num: tl.constexpr,
    cpu_v_start_head_index: tl.constexpr,
    cpu_v_head_num: tl.constexpr,
    gpu_v_start_head_index: tl.constexpr,
    gpu_v_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    block_index_start = tl.program_id(0)
    split_block_num = tl.num_programs(0)
    for block_index in range(block_index_start, copy_block_num, split_block_num):
        token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_mask = token_range < copy_token_num
        gpu_mem_indexes = tl.load(gpu_mem_indexes_ptr + token_range, mask=token_mask).to(tl.int64)
        cpu_mem_indexes = tl.load(cpu_mem_indexes_ptr + token_range, mask=token_mask).to(tl.int64)
        cpu_page_indexes = tl.load(cpu_page_indexes_ptr + token_range, mask=token_mask).to(tl.int64)

        head_dim_range = tl.arange(0, BLOCK_HEAD_DIM)
        head_dim_mask = head_dim_range < head_dim
        scale_head_dim_mask = head_dim_range < scale_head_dim

        for layer_index in range(layer_num):
            move_mask = token_mask[:, None] & head_dim_mask[None, :]
            scale_move_mask = token_mask[:, None] & scale_head_dim_mask[None, :]

            for k_head_index in range(cpu_k_head_num):
                gpu_k_head_index = k_head_index + gpu_k_start_head_index
                cpu_k_head_index = k_head_index + cpu_k_start_head_index

                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_indexes[:, None] * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + cpu_mem_indexes[:, None] * cpu_stride2
                    + cpu_k_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                cpu_data = tl.load(cpu_ptr, mask=move_mask, other=0.0)

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + gpu_mem_indexes[:, None] * gpu_stride1
                    + gpu_k_head_index * gpu_stride2
                    + head_dim_range[None, :]
                )

                tl.store(
                    gpu_ptr,
                    cpu_data,
                    mask=move_mask,
                )
                if HAS_SCALE:
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_indexes[:, None] * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + cpu_mem_indexes[:, None] * cpu_scale_stride2
                        + cpu_k_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    cpu_scale_data = tl.load(cpu_scale_ptr, mask=scale_move_mask, other=0.0)

                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + gpu_mem_indexes[:, None] * gpu_scale_stride1
                        + gpu_k_head_index * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )

                    tl.store(
                        gpu_scale_ptr,
                        cpu_scale_data,
                        mask=scale_move_mask,
                    )

            for v_head_index in range(cpu_v_head_num):
                gpu_v_head_index = v_head_index + gpu_v_start_head_index
                cpu_v_head_index = v_head_index + cpu_v_start_head_index

                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_indexes[:, None] * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + cpu_mem_indexes[:, None] * cpu_stride2
                    + cpu_v_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                cpu_data = tl.load(cpu_ptr, mask=move_mask, other=0.0)

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + gpu_mem_indexes[:, None] * gpu_stride1
                    + gpu_v_head_index * gpu_stride2
                    + head_dim_range[None, :]
                )

                tl.store(
                    gpu_ptr,
                    cpu_data,
                    mask=move_mask,
                )
                if HAS_SCALE:
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_indexes[:, None] * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + cpu_mem_indexes[:, None] * cpu_scale_stride2
                        + cpu_v_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    cpu_scale_data = tl.load(cpu_scale_ptr, mask=scale_move_mask, other=0.0)

                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + gpu_mem_indexes[:, None] * gpu_scale_stride1
                        + gpu_v_head_index * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )

                    tl.store(
                        gpu_scale_ptr,
                        cpu_scale_data,
                        mask=scale_move_mask,
                    )

    return


@torch.no_grad()
def load_cpu_kv_to_gpu(
    gpu_mem_indexes: torch.Tensor,
    gpu_kv_cache: torch.Tensor,
    gpu_kv_cache_scale: Optional[torch.Tensor],
    cpu_kv_cache: torch.Tensor,
    cpu_kv_cache_scale: Optional[torch.Tensor],
    page_indexes: torch.Tensor,
    tp_index: int,
    tp_world_size: int,
    grid_num: int,
    _cache_data={},
):
    """
    this function is used to load CPU KV cache back to GPU KV cache.
    Args:
        gpu_mem_indexes: (token_num,)
        gpu_kv_cache: (layer_num, all_token_num, head_num, head_dim)
        cpu_kv_cache: (all_page_num, layer_num, token_block_size, head_num, head_dim)
        page_indexes: (page_num,)
    """
    token_block_size = cpu_kv_cache.shape[2]
    cpu_page_num = page_indexes.shape[0]
    cpu_page_all_token_num = cpu_page_num * token_block_size
    assert gpu_mem_indexes.shape[0] <= cpu_page_all_token_num
    move_token_num = gpu_mem_indexes.shape[0]

    # NOTE: Avoid GPU kernels directly dereferencing CPU pointers.
    # We first stage the requested CPU pages into a GPU tensor using explicit H2D copies.
    assert page_indexes.device.type == "cpu", "page_indexes must be a CPU tensor in the non-mapped path"

    cpu_mem_indexes = torch.arange(0, cpu_page_all_token_num, device="cuda", dtype=torch.int32) % token_block_size
    cpu_page_indexes_local = (
        torch.arange(0, cpu_page_num, device="cuda", dtype=torch.int32)
        .view((cpu_page_num, 1))
        .tile((1, token_block_size))
        .view(-1)
    )
    cpu_page_indexes_local = cpu_page_indexes_local[-move_token_num:]
    cpu_mem_indexes = cpu_mem_indexes[-move_token_num:]

    gpu_heads = gpu_kv_cache.shape[2]
    gpu_head_dim = gpu_kv_cache.shape[3]
    cpu_heads = cpu_kv_cache.shape[3]
    cpu_head_dim = cpu_kv_cache.shape[4]
    head_dim = gpu_head_dim
    scale_size = (tp_world_size * gpu_heads) // cpu_heads

    # 计算需要拷贝的 head 索引的对应关系
    if (gpu_heads, cpu_heads, tp_index, tp_world_size) in _cache_data:
        head_info_tuple = _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)]
    else:
        if cpu_heads > 1:
            assert (tp_world_size * gpu_heads) % cpu_heads == 0
            assert cpu_heads % 2 == 0

            cpu_heads_index = (
                torch.arange(0, cpu_heads, device="cpu", dtype=torch.int32)
                .view(cpu_heads, 1)
                .tile((1, scale_size))
                .view(2, tp_world_size, -1)
            )
            # k
            k_cpu_heads_index = cpu_heads_index[0][tp_index]
            # v
            v_cpu_heads_index = cpu_heads_index[1][tp_index]

            cpu_heads_index = torch.cat([k_cpu_heads_index, v_cpu_heads_index], dim=0).view(2, -1).numpy()
            gpu_heads_index = torch.arange(0, gpu_heads, device="cpu", dtype=torch.int32).view(2, -1).numpy()

            cpu_k_start_head_index = int(cpu_heads_index[0, 0])
            cpu_k_head_num = len(cpu_heads_index[0])
            gpu_k_start_head_index = int(gpu_heads_index[0, 0])
            gpu_k_head_num = len(gpu_heads_index[0])
            assert cpu_k_head_num == gpu_k_head_num
            cpu_v_start_head_index = int(cpu_heads_index[1, 0])
            cpu_v_head_num = len(cpu_heads_index[1])
            gpu_v_start_head_index = int(gpu_heads_index[1, 0])
            gpu_v_head_num = len(gpu_heads_index[1])
            assert cpu_v_head_num == gpu_v_head_num

            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        else:
            assert gpu_heads == 1
            assert cpu_heads == 1

            cpu_k_start_head_index = 0
            cpu_k_head_num = 1
            gpu_k_start_head_index = 0
            gpu_k_head_num = 1
            cpu_v_start_head_index = 0
            cpu_v_head_num = 0
            gpu_v_start_head_index = 0
            gpu_v_head_num = 0
            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)] = head_info_tuple

    (
        cpu_k_start_head_index,
        cpu_k_head_num,
        gpu_k_start_head_index,
        gpu_k_head_num,
        cpu_v_start_head_index,
        cpu_v_head_num,
        gpu_v_start_head_index,
        gpu_v_head_num,
    ) = head_info_tuple

    TOKEN_BLOCK = 128

    grid = (grid_num,)
    num_warps = 4

    HAS_SCALE_TENSOR = gpu_kv_cache_scale is not None and cpu_kv_cache_scale is not None
    HAS_MERGED_SCALE = gpu_kv_cache_scale is not None and (not HAS_SCALE_TENSOR) and (cpu_head_dim > head_dim)

    if gpu_kv_cache_scale is None:
        assert cpu_head_dim == head_dim

    if HAS_MERGED_SCALE:
        assert cpu_kv_cache.dtype in (
            torch.int8,
            torch.uint8,
        ), "merged KV+scale requires byte-addressable cpu_kv_cache dtype"
        scale_head_dim = gpu_kv_cache_scale.shape[-1]
        gpu_scale_stride = gpu_kv_cache_scale.stride()
        scale_bytes_per_head = scale_head_dim * gpu_kv_cache_scale.element_size()
        kv_bytes_per_head = head_dim * cpu_kv_cache.element_size()
        merged_bytes_per_head = cpu_head_dim * cpu_kv_cache.element_size()
        assert (
            kv_bytes_per_head + scale_bytes_per_head == merged_bytes_per_head
        ), "cpu merged head dim bytes must equal kv bytes + scale bytes"

        # Stage full merged pages into GPU (one H2D per page).
        gpu_staging_merged = torch.empty(
            (cpu_page_num, cpu_kv_cache.shape[1], token_block_size, cpu_heads, cpu_head_dim),
            device=gpu_kv_cache.device,
            dtype=cpu_kv_cache.dtype,
        )
        gpu_staging = gpu_staging_merged[..., :head_dim]
        gpu_staging_scale = gpu_staging_merged[..., head_dim:].view(gpu_kv_cache_scale.dtype)
    else:
        if HAS_SCALE_TENSOR:
            scale_head_dim = gpu_kv_cache_scale.shape[-1]
            gpu_scale_stride = gpu_kv_cache_scale.stride()
            scale_page_dtype = cpu_kv_cache_scale.dtype
        else:
            scale_head_dim = 0
            gpu_scale_stride = [0 for _ in range(10)]

        gpu_staging = torch.empty(
            (cpu_page_num, cpu_kv_cache.shape[1], token_block_size, cpu_heads, head_dim),
            device=gpu_kv_cache.device,
            dtype=cpu_kv_cache.dtype,
        )
        if HAS_SCALE_TENSOR:
            gpu_staging_scale = torch.empty(
                (cpu_page_num, cpu_kv_cache_scale.shape[1], token_block_size, cpu_heads, scale_head_dim),
                device=gpu_kv_cache_scale.device,
                dtype=scale_page_dtype,
            )
        else:
            gpu_staging_scale = None

    stream = torch.cuda.current_stream().cuda_stream
    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    for i in range(cpu_page_num):
        src_page = int(page_indexes[i].item())
        if src_page < 0 or src_page >= cpu_kv_cache.shape[0]:
            raise RuntimeError(f"cpu_kv_cache page index out of range: {src_page}, page_num={cpu_kv_cache.shape[0]}")
        src_view = cpu_kv_cache[src_page]
        dst_view = gpu_staging_merged[i] if HAS_MERGED_SCALE else gpu_staging[i]
        assert src_view.is_contiguous(), "cpu_kv_cache page view must be contiguous"
        assert dst_view.is_contiguous(), "gpu staging must be contiguous"
        _cuda_memcpy_async(
            dst_ptr=dst_view.data_ptr(),
            src_ptr=src_view.data_ptr(),
            nbytes=dst_view.numel() * dst_view.element_size(),
            kind=CUDA_MEMCPY_HOST_TO_DEVICE,
            stream=stream,
        )
        if HAS_SCALE_TENSOR:
            src_scale_view = cpu_kv_cache_scale[src_page]
            dst_scale_view = gpu_staging_scale[i]
            assert src_scale_view.is_contiguous(), "cpu_kv_cache_scale page view must be contiguous"
            assert dst_scale_view.is_contiguous(), "gpu scale staging must be contiguous"
            _cuda_memcpy_async(
                dst_ptr=dst_scale_view.data_ptr(),
                src_ptr=src_scale_view.data_ptr(),
                nbytes=dst_scale_view.numel() * dst_scale_view.element_size(),
                kind=CUDA_MEMCPY_HOST_TO_DEVICE,
                stream=stream,
            )

    _load_cpu_cache_to_gpu[grid](
        gpu_mem_indexes_ptr=gpu_mem_indexes,
        copy_token_num=move_token_num,
        copy_block_num=triton.cdiv(move_token_num, TOKEN_BLOCK),
        cpu_mem_indexes_ptr=cpu_mem_indexes,
        cpu_page_indexes_ptr=cpu_page_indexes_local,
        gpu_kv_cache_ptr=gpu_kv_cache,
        gpu_stride0=gpu_kv_cache.stride(0),
        gpu_stride1=gpu_kv_cache.stride(1),
        gpu_stride2=gpu_kv_cache.stride(2),
        gpu_stride3=gpu_kv_cache.stride(3),
        gpu_kv_cache_scale_ptr=gpu_kv_cache_scale,
        gpu_scale_stride0=gpu_scale_stride[0],
        gpu_scale_stride1=gpu_scale_stride[1],
        gpu_scale_stride2=gpu_scale_stride[2],
        gpu_scale_stride3=gpu_scale_stride[3],
        cpu_kv_cache_ptr=gpu_staging,
        cpu_stride0=gpu_staging.stride(0),
        cpu_stride1=gpu_staging.stride(1),
        cpu_stride2=gpu_staging.stride(2),
        cpu_stride3=gpu_staging.stride(3),
        cpu_stride4=gpu_staging.stride(4),
        cpu_kv_cache_scale_ptr=gpu_staging_scale,
        cpu_scale_stride0=(gpu_staging_scale.stride(0) if gpu_staging_scale is not None else 0),
        cpu_scale_stride1=(gpu_staging_scale.stride(1) if gpu_staging_scale is not None else 0),
        cpu_scale_stride2=(gpu_staging_scale.stride(2) if gpu_staging_scale is not None else 0),
        cpu_scale_stride3=(gpu_staging_scale.stride(3) if gpu_staging_scale is not None else 0),
        cpu_scale_stride4=(gpu_staging_scale.stride(4) if gpu_staging_scale is not None else 0),
        layer_num=gpu_kv_cache.shape[0],
        head_dim=head_dim,
        scale_head_dim=scale_head_dim,
        cpu_k_start_head_index=cpu_k_start_head_index,
        cpu_k_head_num=cpu_k_head_num,
        gpu_k_start_head_index=gpu_k_start_head_index,
        gpu_k_head_num=gpu_k_head_num,
        cpu_v_start_head_index=cpu_v_start_head_index,
        cpu_v_head_num=cpu_v_head_num,
        gpu_v_start_head_index=gpu_v_start_head_index,
        gpu_v_head_num=gpu_v_head_num,
        BLOCK_HEAD_DIM=triton.next_power_of_2(head_dim),
        TOKEN_BLOCK=TOKEN_BLOCK,
        HAS_SCALE=(gpu_staging_scale is not None),
        num_warps=num_warps,
        num_stages=1,
    )
    return
