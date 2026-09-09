import torch
import triton
import triton.language as tl


@triton.jit
def _copy_sliding_window_cpu_cache(
    mem_indexes,
    page_indexes,
    page_readies,
    big_page_buffer_ids,
    gpu_full_att_kv_state,
    cpu_kv_sliding_state,
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
                state_ptr = cpu_kv_sliding_state + big_page * WINDOW_RANK_SIZE + offsets
                cpu_ptr = cpu_cache + cpu_page_start + FULL_TOTAL_SIZE + tp_rank * WINDOW_RANK_SIZE + offsets
                if OFFLOAD:
                    value = tl.load(state_ptr, valid, other=0)
                    tl.store(cpu_ptr, value, valid)
                else:
                    value = tl.load(cpu_ptr, valid, other=0)
                    tl.store(state_ptr, value, valid)


def _copy_state_cache(
    mem_indexes,
    page_indexes,
    page_readies,
    big_page_buffer_ids,
    gpu_full_att_kv_state,
    cpu_kv_sliding_state,
    cpu_cache_tensor,
    tp_rank,
    tp_world_size,
    big_page_token_num,
    offload,
    grid_num,
):
    page_num = len(page_indexes)
    assert len(big_page_buffer_ids) == page_num
    assert len(mem_indexes) == page_num * big_page_token_num
    assert not offload or len(page_readies) == page_num
    if page_num == 0:
        return

    assert gpu_full_att_kv_state.is_contiguous() and cpu_kv_sliding_state.is_contiguous()
    assert cpu_cache_tensor.is_contiguous()

    # Packing preserves the original bit patterns, including BF16/FP16 NaNs.
    # Storage dimensions, not a model config, define each rank's payload.
    # view(uint64) also checks the required element-size alignment.
    full_state = gpu_full_att_kv_state.flatten(2).view(torch.uint64)
    window_state = cpu_kv_sliding_state.flatten(1).view(torch.uint64)
    cpu_cache = cpu_cache_tensor.view(cpu_cache_tensor.shape[0], -1).view(torch.uint64)
    full_rank_size = big_page_token_num * full_state.shape[0] * full_state.shape[2]
    window_rank_size = window_state.shape[1]
    page_size = (full_rank_size + window_rank_size) * tp_world_size
    assert (
        cpu_cache.shape[1] == triton.cdiv(page_size, 2) * 2
    ), "CPU byte-page layout does not match GPU/checkpoint storage"

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
        FULL_LAYER_NUM=full_state.shape[0],
        FULL_TOKEN_LAYER_SIZE=full_state.shape[2],
        FULL_RANK_SIZE=full_rank_size,
        FULL_TOTAL_SIZE=full_rank_size * tp_world_size,
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
    cpu_kv_sliding_state: torch.Tensor,
    cpu_cache_tensor: torch.Tensor,
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    grid_num: int = 12,
):
    """Pack GPU full KV and pinned CPU checkpoints into this TP rank's CPU page slices."""
    _copy_state_cache(
        mem_indexes,
        page_indexes,
        page_readies,
        big_page_buffer_ids,
        gpu_full_att_kv_state,
        cpu_kv_sliding_state,
        cpu_cache_tensor,
        tp_rank,
        tp_world_size,
        big_page_token_num,
        offload=True,
        grid_num=grid_num,
    )


def copy_cpu_cache_to_kv_buffer(
    mem_indexes: torch.Tensor,
    page_indexes: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_full_att_kv_state: torch.Tensor,
    cpu_kv_sliding_state: torch.Tensor,
    cpu_cache_tensor: torch.Tensor,
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    grid_num: int = 12,
):
    """Restore GPU full KV and pinned CPU checkpoints from a packed CPU page."""
    _copy_state_cache(
        mem_indexes,
        page_indexes,
        None,
        big_page_buffer_ids,
        gpu_full_att_kv_state,
        cpu_kv_sliding_state,
        cpu_cache_tensor,
        tp_rank,
        tp_world_size,
        big_page_token_num,
        offload=False,
        grid_num=grid_num,
    )


@triton.jit
def _copy_sliding_window_state(
    Src,
    Dst,
    src_layer_stride,
    dst_layer_stride,
    LAYER_BYTES: tl.constexpr,
    TOTAL_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    for block in range(pid, tl.cdiv(TOTAL_BYTES, BLOCK), tl.num_programs(0)):
        offsets = tl.cast(block, tl.int64) * BLOCK + tl.arange(0, BLOCK)
        layer = offsets // LAYER_BYTES
        within_layer = offsets % LAYER_BYTES
        src = Src + layer * tl.cast(src_layer_stride, tl.int64) + within_layer
        dst = Dst + layer * tl.cast(dst_layer_stride, tl.int64) + within_layer
        values = tl.load(src, mask=offsets < TOTAL_BYTES, other=0)
        tl.store(dst, values, mask=offsets < TOTAL_BYTES)


@torch.no_grad()
def copy_sliding_window_state(src_state: torch.Tensor, dst_state: torch.Tensor):
    """Copy [layer, ...payload] on the CUDA stream, including pinned CPU staging."""
    assert src_state.shape == dst_state.shape
    assert src_state.dtype == dst_state.dtype
    if not src_state.numel():
        return
    assert src_state[0].is_contiguous() and dst_state[0].is_contiguous()
    # Only the layer stride may contain gaps in a request's GPU runtime view.
    # Reinterpret bytes without flattening/copying that noncontiguous view.
    src_bytes, dst_bytes = src_state.view(torch.uint8), dst_state.view(torch.uint8)
    total_bytes = src_bytes.numel()
    _copy_sliding_window_state[(16,)](
        src_bytes,
        dst_bytes,
        src_bytes.stride(0),
        dst_bytes.stride(0),
        LAYER_BYTES=total_bytes // src_state.shape[0],
        TOTAL_BYTES=total_bytes,
        BLOCK=4096,
    )
