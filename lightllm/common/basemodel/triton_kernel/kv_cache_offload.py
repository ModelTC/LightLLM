import torch

import triton
import triton.language as tl


@triton.jit
def _offload_gpu_kv_to_cpu(
    token_indexes_ptr,
    gpu_kv_cache_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    gpu_stride3,
    cpu_kv_cache_ptr,
    cpu_stride0,
    cpu_stride1,
    cpu_stride2,
    cpu_stride3,
    cpu_stride4,
    page_indexes_ptr,
    page_readies_ptr,
    layer_num,
    head_all_dim,
    cpu_head_offset,
    BLOCK_HEAD_ALL_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    block_index = tl.program_id(0)
    cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)
    if cpu_page_index == -1:
        return

    ready_state = tl.load(page_readies_ptr + block_index)
    if ready_state:
        return

    token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    token_indexes = tl.load(token_indexes_ptr + token_range).to(tl.int64)
    head_all_dim_range = tl.arange(0, BLOCK_HEAD_ALL_DIM)

    gpu_stride0 = tl.cast(gpu_stride0, dtype=tl.int64)

    for layer_index in range(layer_num):
        gpu_ptr = (
            gpu_kv_cache_ptr
            + layer_index * gpu_stride0
            + token_indexes[:, None] * gpu_stride1
            + head_all_dim_range[None, :]
        )
        gpu_data = tl.load(gpu_ptr, mask=(head_all_dim_range[None, :] < head_all_dim), other=0.0)
        gpu_data = tl.zeros_like(gpu_data)
        cpu_ptr = (
            cpu_kv_cache_ptr
            + cpu_page_index
            + layer_index
            + tl.arange(0, TOKEN_BLOCK)[:, None] * 0
            + (cpu_head_offset + head_all_dim_range[None, :]) * 0
        )
        tl.store(
            cpu_ptr,
            gpu_data,
            mask=(head_all_dim_range[None, :] < head_all_dim),
        )
    return


@torch.no_grad()
def offload_gpu_kv_to_cpu(
    token_indexes: torch.Tensor,
    gpu_kv_cache: torch.Tensor,
    cpu_kv_cache: torch.Tensor,
    page_indexes: torch.Tensor,
    page_readies: torch.Tensor,
):
    """
    this function is used to offload GPU KV cache to CPU KV cache.
    Supports tensor parallelism (TP > 1).
    Args:
        token_indexes: (token_num,)
        gpu_kv_cache: (layer_num, token_num, head_num, head_dim)
        cpu_kv_cache: (all_page_num, layer_num, token_block_size, head_num, head_dim)
        page_indexes: (page_num,)
        page_readies: (page_num,)
    """
    from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
    
    token_block_size = cpu_kv_cache.shape[2]
    token_num = page_indexes.shape[0] * token_block_size
    assert token_indexes.shape[0] >= token_num
    assert page_indexes.shape == page_readies.shape
    page_num = page_indexes.shape[0]
    head_all_dim = gpu_kv_cache.shape[-1] * gpu_kv_cache.shape[-2]
    BLOCK_HEAD_ALL_DIM = triton.next_power_of_2(gpu_kv_cache.shape[-1] * gpu_kv_cache.shape[-2])

    # Calculate head offset for tensor parallelism
    tp_rank = get_current_rank_in_dp()
    gpu_heads = gpu_kv_cache.shape[2]
    gpu_head_dim = gpu_kv_cache.shape[3]
    cpu_head_offset = tp_rank * gpu_heads * gpu_head_dim

    grid = (page_num,)
    num_warps = 4

    _offload_gpu_kv_to_cpu[grid](
        token_indexes_ptr=token_indexes,
        gpu_kv_cache_ptr=gpu_kv_cache,
        gpu_stride0=gpu_kv_cache.stride(0),
        gpu_stride1=gpu_kv_cache.stride(1),
        gpu_stride2=gpu_kv_cache.stride(2),
        gpu_stride3=gpu_kv_cache.stride(3),
        cpu_kv_cache_ptr=cpu_kv_cache,
        cpu_stride0=cpu_kv_cache.stride(0),
        cpu_stride1=cpu_kv_cache.stride(1),
        cpu_stride2=cpu_kv_cache.stride(2),
        cpu_stride3=cpu_kv_cache.stride(3),
        cpu_stride4=cpu_kv_cache.stride(4),
        page_indexes_ptr=page_indexes,
        page_readies_ptr=page_readies,
        layer_num=gpu_kv_cache.shape[0],
        head_all_dim=head_all_dim,
        cpu_head_offset=cpu_head_offset,
        BLOCK_HEAD_ALL_DIM=BLOCK_HEAD_ALL_DIM,
        TOKEN_BLOCK=token_block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _load_cpu_cache_to_gpu(
    token_indexes_ptr,
    gpu_kv_cache_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    gpu_stride3,
    cpu_kv_cache_ptr,
    cpu_stride0,
    cpu_stride1,
    cpu_stride2,
    cpu_stride3,
    cpu_stride4,
    page_indexes_ptr,
    layer_num,
    head_all_dim,
    all_move_token_num,
    cpu_head_offset,
    BLOCK_HEAD_ALL_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    block_index = tl.program_id(0)
    cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)
    if cpu_page_index == -1:
        return

    gpu_stride0 = tl.cast(gpu_stride0, dtype=tl.int64)
    padded_size = TOKEN_BLOCK * tl.num_programs(0) - all_move_token_num
    head_all_dim_range = tl.arange(0, BLOCK_HEAD_ALL_DIM)
    token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    token_range = token_range - padded_size

    token_mask = token_range >= 0
    head_dim_mask = head_all_dim_range < head_all_dim

    token_indexes = tl.load(token_indexes_ptr + token_range, mask=token_mask, other=0).to(tl.int64)

    cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)
    for layer_index in range(layer_num):
        cpu_ptr = (
            cpu_kv_cache_ptr
            + cpu_page_index * cpu_stride0
            + layer_index * cpu_stride1
            + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_stride2
            + (cpu_head_offset + head_all_dim_range[None, :])
        )
        cpu_data = tl.load(cpu_ptr, mask=head_dim_mask[None, :], other=0.0)

        gpu_ptr = (
            gpu_kv_cache_ptr
            + layer_index * gpu_stride0
            + token_indexes[:, None] * gpu_stride1
            + head_all_dim_range[None, :]
        )
        tl.store(
            gpu_ptr,
            cpu_data,
            mask=token_mask[:, None] & head_dim_mask[None, :],
        )
    return


@torch.no_grad()
def load_cpu_kv_to_gpu(
    mem_indexes: torch.Tensor,
    gpu_kv_cache: torch.Tensor,
    cpu_kv_cache: torch.Tensor,
    page_indexes: torch.Tensor,
):
    """
    this function is used to load CPU KV cache to GPU KV cache.
    Supports tensor parallelism (TP > 1).
    Args:
        mem_indexes: (token_num,)
        gpu_kv_cache: (layer_num, token_num, head_num, head_dim)
        cpu_kv_cache: (page_num, layer_num, token_block_size, head_num, head_dim)
        page_indexes: (page_num,)
    """
    from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
    
    token_block_size = cpu_kv_cache.shape[2]
    token_num = page_indexes.shape[0] * token_block_size
    assert mem_indexes.shape[0] >= token_num
    page_num = page_indexes.shape[0]
    BLOCK_HEAD_ALL_DIM = triton.next_power_of_2(gpu_kv_cache.shape[-1] * gpu_kv_cache.shape[-2])

    # Calculate head offset for tensor parallelism
    tp_rank = get_current_rank_in_dp()
    gpu_heads = gpu_kv_cache.shape[2]
    gpu_head_dim = gpu_kv_cache.shape[3]
    cpu_head_offset = tp_rank * gpu_heads * gpu_head_dim

    grid = (page_num,)
    num_warps = 1

    _load_cpu_cache_to_gpu[grid](
        token_indexes_ptr=mem_indexes,
        gpu_kv_cache_ptr=gpu_kv_cache,
        gpu_stride0=gpu_kv_cache.stride(0),
        gpu_stride1=gpu_kv_cache.stride(1),
        gpu_stride2=gpu_kv_cache.stride(2),
        gpu_stride3=gpu_kv_cache.stride(3),
        cpu_kv_cache_ptr=cpu_kv_cache,
        cpu_stride0=cpu_kv_cache.stride(0),
        cpu_stride1=cpu_kv_cache.stride(1),
        cpu_stride2=cpu_kv_cache.stride(2),
        cpu_stride3=cpu_kv_cache.stride(3),
        cpu_stride4=cpu_kv_cache.stride(4),
        page_indexes_ptr=page_indexes,
        layer_num=gpu_kv_cache.shape[0],
        head_all_dim=gpu_kv_cache.shape[-1] * gpu_kv_cache.shape[-2],
        all_move_token_num=len(mem_indexes),
        cpu_head_offset=cpu_head_offset,
        BLOCK_HEAD_ALL_DIM=BLOCK_HEAD_ALL_DIM,
        TOKEN_BLOCK=token_block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return
