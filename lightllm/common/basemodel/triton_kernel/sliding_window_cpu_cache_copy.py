import torch
import triton
import triton.language as tl

from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig


@triton.jit
def _copy_sliding_window_cpu_cache(
    mem_indexes,
    page_indexes,
    page_readies,
    big_page_buffer_ids,
    gpu_full_att_kv_state,
    gpu_sliding_state,
    cpu_cache,
    page_num,
    full_stride_l,
    full_stride_t,
    cpu_stride_p,
    tp_rank,
    FULL_LAYER_NUM: tl.constexpr,
    FULL_TOKEN_LAYER_SIZE: tl.constexpr,
    FULL_RANK_SIZE: tl.constexpr,
    FULL_TOTAL_SIZE: tl.constexpr,
    WINDOW_RANK_SIZE: tl.constexpr,
    BIG_PAGE_TOKEN_NUM: tl.constexpr,
    OFFLOAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # All sizes/offsets below are uint64 elements, not bytes. CPU page offsets
    # must remain int64: a shared CPU cache can be much larger than 2 GiB.
    full_stride_l = tl.cast(full_stride_l, tl.int64)
    full_stride_t = tl.cast(full_stride_t, tl.int64)
    cpu_stride_p = tl.cast(cpu_stride_p, tl.int64)
    tp_rank = tl.cast(tp_rank, tl.int64)
    block_start = tl.program_id(0)
    block_count = tl.num_programs(0)
    for page in range(page_num):
        cpu_page = tl.load(page_indexes + page).to(tl.int64)
        copy_page = cpu_page != -1
        if OFFLOAD:
            copy_page = copy_page & ~tl.load(page_readies + page).to(tl.int1)
        if copy_page:
            cpu_page_start = cpu_page * cpu_stride_p
            for block in range(block_start, tl.cdiv(FULL_RANK_SIZE, BLOCK), block_count):
                offsets = tl.cast(block, tl.int64) * BLOCK + tl.arange(0, BLOCK)
                valid = offsets < FULL_RANK_SIZE
                token = offsets // (FULL_LAYER_NUM * FULL_TOKEN_LAYER_SIZE)
                layer = (offsets // FULL_TOKEN_LAYER_SIZE) % FULL_LAYER_NUM
                dim = offsets % FULL_TOKEN_LAYER_SIZE
                mem_index = tl.load(mem_indexes + page * BIG_PAGE_TOKEN_NUM + token, valid, other=-1).to(tl.int64)
                valid = valid & (mem_index != -1)
                gpu_ptr = gpu_full_att_kv_state + layer * full_stride_l + mem_index * full_stride_t + dim
                cpu_ptr = cpu_cache + cpu_page_start + tp_rank * FULL_RANK_SIZE + offsets
                if OFFLOAD:
                    value = tl.load(gpu_ptr, valid, other=0)
                    tl.store(cpu_ptr, value, valid)
                else:
                    value = tl.load(cpu_ptr, valid, other=0)
                    tl.store(gpu_ptr, value, valid)

            big_page = tl.load(big_page_buffer_ids + page).to(tl.int64)
            for block in range(block_start, tl.cdiv(WINDOW_RANK_SIZE, BLOCK), block_count):
                offsets = tl.cast(block, tl.int64) * BLOCK + tl.arange(0, BLOCK)
                valid = offsets < WINDOW_RANK_SIZE
                gpu_ptr = gpu_sliding_state + big_page * WINDOW_RANK_SIZE + offsets
                cpu_ptr = cpu_cache + cpu_page_start + FULL_TOTAL_SIZE + tp_rank * WINDOW_RANK_SIZE + offsets
                if OFFLOAD:
                    value = tl.load(gpu_ptr, valid, other=0)
                    tl.store(cpu_ptr, value, valid)
                else:
                    value = tl.load(cpu_ptr, valid, other=0)
                    tl.store(gpu_ptr, value, valid)


def _copy_state_cache(
    mem_indexes,
    page_indexes,
    page_readies,
    big_page_buffer_ids,
    gpu_full_att_kv_state,
    gpu_sliding_state,
    cpu_cache_tensor,
    tp_rank,
    tp_world_size,
    big_page_token_num,
    sliding_config,
    offload,
    grid_num,
):
    page_num = len(page_indexes)
    assert len(big_page_buffer_ids) == page_num
    assert len(mem_indexes) == page_num * big_page_token_num
    assert not offload or len(page_readies) == page_num
    assert 0 <= tp_rank < tp_world_size
    assert big_page_token_num > 0 and grid_num > 0
    if page_num == 0:
        return

    assert gpu_full_att_kv_state.shape[0] == sliding_config.full_layer_num
    assert gpu_full_att_kv_state.shape[2:] == (
        2 * sliding_config.full_head_num,
        sliding_config.full_head_dim,
    )
    assert gpu_sliding_state.shape[1:] == sliding_config.get_state_shape()
    assert gpu_full_att_kv_state.dtype == gpu_sliding_state.dtype == sliding_config.dtype
    assert gpu_full_att_kv_state.is_contiguous() and gpu_sliding_state.is_contiguous()
    assert cpu_cache_tensor.is_contiguous()

    cpu_cache = cpu_cache_tensor.view(cpu_cache_tensor.shape[0], -1).view(torch.uint8)
    full_bytes = sliding_config.get_cpu_cache_full_att_bytes(big_page_token_num, tp_world_size)
    window_bytes = sliding_config.get_cpu_cache_state_bytes(tp_world_size)
    assert cpu_cache.shape[1] == sliding_config.get_cpu_cache_big_page_bytes(big_page_token_num, tp_world_size)
    # Packing preserves the original bit patterns, including BF16/FP16 NaNs.
    # Gemma's K+V head rows and checkpoint tensors are all uint64-aligned.
    full_state = gpu_full_att_kv_state.flatten(2).view(torch.uint64)
    window_state = gpu_sliding_state.flatten(1).view(torch.uint64)
    cpu_cache = cpu_cache.view(torch.uint64)
    full_rank_size = big_page_token_num * full_state.shape[0] * full_state.shape[2]
    window_rank_size = window_state.shape[1]
    assert full_rank_size * tp_world_size * 8 == full_bytes
    assert window_rank_size * tp_world_size * 8 == window_bytes

    _copy_sliding_window_cpu_cache[(grid_num,)](
        mem_indexes,
        page_indexes,
        page_readies,
        big_page_buffer_ids,
        full_state,
        window_state,
        cpu_cache,
        page_num,
        full_state.stride(0),
        full_state.stride(1),
        cpu_cache.stride(0),
        tp_rank,
        FULL_LAYER_NUM=sliding_config.full_layer_num,
        FULL_TOKEN_LAYER_SIZE=full_state.shape[2],
        FULL_RANK_SIZE=full_rank_size,
        FULL_TOTAL_SIZE=full_bytes // 8,
        WINDOW_RANK_SIZE=window_rank_size,
        BIG_PAGE_TOKEN_NUM=big_page_token_num,
        OFFLOAD=offload,
        BLOCK=4096,
    )


def copy_kv_buffer_to_cpu_cache(
    mem_indexes: torch.Tensor,
    page_indexes: torch.Tensor,
    page_readies: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_full_att_kv_state: torch.Tensor,
    gpu_sliding_state: torch.Tensor,
    cpu_cache_tensor: torch.Tensor,
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    sliding_config: SlidingWindowCacheConfig,
    grid_num: int = 12,
):
    """Pack full KV and a raw ring checkpoint into this TP rank's CPU page slices."""
    _copy_state_cache(
        mem_indexes,
        page_indexes,
        page_readies,
        big_page_buffer_ids,
        gpu_full_att_kv_state,
        gpu_sliding_state,
        cpu_cache_tensor,
        tp_rank,
        tp_world_size,
        big_page_token_num,
        sliding_config,
        offload=True,
        grid_num=grid_num,
    )


def copy_cpu_cache_to_kv_buffer(
    mem_indexes: torch.Tensor,
    page_indexes: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_full_att_kv_state: torch.Tensor,
    gpu_sliding_state: torch.Tensor,
    cpu_cache_tensor: torch.Tensor,
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    sliding_config: SlidingWindowCacheConfig,
    grid_num: int = 12,
):
    """Restore full KV and the unchanged ring checkpoint from a packed CPU page."""
    _copy_state_cache(
        mem_indexes,
        page_indexes,
        None,
        big_page_buffer_ids,
        gpu_full_att_kv_state,
        gpu_sliding_state,
        cpu_cache_tensor,
        tp_rank,
        tp_world_size,
        big_page_token_num,
        sliding_config,
        offload=False,
        grid_num=grid_num,
    )
