import torch
import triton
import triton.language as tl

from .swa_cache import get_swa_token_index


@triton.jit
def _build_image_visibility_kernel(
    image_spans,  # (num_images, 3): batch index, absolute image start, visible image length
    b_q_start_loc,
    b_ready_cache_len,
    b_q_seq_len,
    image_left,
    image_right,
    BLOCK_SIZE: tl.constexpr,
):
    image_idx = tl.program_id(0)
    span_ptr = image_spans + image_idx * 3
    batch_idx = tl.load(span_ptr)
    image_start = tl.load(span_ptr + 1)
    image_len = tl.load(span_ptr + 2)
    cache_len = tl.load(b_ready_cache_len + batch_idx)
    q_seq_len = tl.load(b_q_seq_len + batch_idx)
    flat_image_start = tl.load(b_q_start_loc + batch_idx) + image_start - cache_len

    for start in range(0, image_len, BLOCK_SIZE):
        offset = start + tl.arange(0, BLOCK_SIZE)
        q_offset = image_start - cache_len + offset
        mask = (offset < image_len) & (q_offset >= 0) & (q_offset < q_seq_len)
        output_offset = flat_image_start + offset
        tl.store(image_left + output_offset, offset, mask=mask)
        tl.store(image_right + output_offset, image_len - 1 - offset, mask=mask)


def build_image_visibility(
    image_spans: torch.Tensor,
    b_q_start_loc: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_q_seq_len: torch.Tensor,
    image_left: torch.Tensor,
    image_right: torch.Tensor,
):
    """Scatter image-relative left/right distances into the flattened prefill layout."""
    _build_image_visibility_kernel[(image_spans.shape[0],)](
        image_spans,
        b_q_start_loc,
        b_ready_cache_len,
        b_q_seq_len,
        image_left,
        image_right,
        BLOCK_SIZE=64,
    )


@triton.jit
def _build_swa_indices_kernel(
    token_req_idx,
    position_ids,
    prefill_token_ranges,
    swa_prefill_workspace_offsets,
    image_left,
    image_right,
    swa_indices,
    swa_indices_stride0,
    swa_lengths,
    swa_write_slots,
    swa_runtime_write_slots,
    token_to_batch_idx,
    b_seq_len,
    WRITE_SWA_RUNTIME: tl.constexpr,
    SWA_PAGE_TOKEN_NUM: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_IMAGE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SWA_CAPACITY_TOKENS: tl.constexpr,
    HOLD_REQUEST_ID: tl.constexpr,
    HOLD_SWA_SLOT: tl.constexpr,
    IS_PREFILL: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx = tl.load(token_req_idx + token_idx).to(tl.int64)
    pos = tl.load(position_ids + token_idx).to(tl.int64)

    w = tl.arange(0, BLOCK_W)
    w_mask = w < WIDTH
    offset = pos - w
    valid = (offset >= 0) & (w < SLIDING_WINDOW)
    length = tl.minimum(tl.maximum(pos + 1, 1), SLIDING_WINDOW)
    if HAS_IMAGE:
        left = tl.load(image_left + token_idx)
        right = tl.load(image_right + token_idx)
        is_image = (left > 0) | (right > 0)
        image_start = tl.maximum(pos - (SLIDING_WINDOW - 1) - tl.maximum(left - (SLIDING_WINDOW - 1), 0), 0)
        image_length = pos + right - image_start + 1
        offset = tl.where(is_image, image_start + w, offset)
        valid = tl.where(is_image, w < image_length, valid)
        length = tl.where(is_image, image_length, length)
    safe_offset = tl.where(valid, offset, 0)
    swa_slot = get_swa_token_index(
        req_idx,
        safe_offset,
        prefill_token_ranges,
        swa_prefill_workspace_offsets,
        SWA_CAPACITY_TOKENS,
        SLIDING_WINDOW,
        SWA_PAGE_TOKEN_NUM,
        HOLD_REQUEST_ID,
        HOLD_SWA_SLOT,
        IS_PREFILL,
    )
    out = tl.where(valid, swa_slot, -1).to(tl.int32)
    tl.store(swa_indices + token_idx * swa_indices_stride0 + w, out, mask=w_mask)

    tl.store(swa_lengths + token_idx, length.to(tl.int32))
    write_slot = get_swa_token_index(
        req_idx,
        pos,
        prefill_token_ranges,
        swa_prefill_workspace_offsets,
        SWA_CAPACITY_TOKENS,
        SLIDING_WINDOW,
        SWA_PAGE_TOKEN_NUM,
        HOLD_REQUEST_ID,
        HOLD_SWA_SLOT,
        IS_PREFILL,
    )
    tl.store(swa_write_slots + token_idx, write_slot)
    if WRITE_SWA_RUNTIME:
        batch = tl.load(token_to_batch_idx + token_idx)
        end = tl.load(b_seq_len + batch)
        keep = (req_idx != HOLD_REQUEST_ID) & (pos >= end - SWA_CAPACITY_TOKENS)
        runtime_slot = req_idx * SWA_CAPACITY_TOKENS + pos % SWA_CAPACITY_TOKENS
        tl.store(swa_runtime_write_slots + token_idx, tl.where(keep, runtime_slot, -1))


def build_swa_indices(
    token_req_idx: torch.Tensor,
    position_ids: torch.Tensor,
    req_manager,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    swa_write_slots: torch.Tensor,
    sliding_window: int = None,
    image_left: torch.Tensor = None,
    image_right: torch.Tensor = None,
    prefill_state=None,
    swa_runtime_write_slots: torch.Tensor = None,
    token_to_batch_idx: torch.Tensor = None,
    b_seq_len: torch.Tensor = None,
):
    """生成 SWA 物理索引；token_req_idx 和 position_ids 都按 query token 展开。"""
    token_num = position_ids.shape[0]
    width = swa_indices.shape[1]
    sliding_window = width if sliding_window is None else int(sliding_window)
    assert sliding_window <= width
    assert (image_left is None) == (image_right is None)
    if token_num == 0:
        return swa_indices, swa_lengths
    image_left_arg = position_ids if image_left is None else image_left
    image_right_arg = position_ids if image_right is None else image_right
    _build_swa_indices_kernel[(token_num,)](
        token_req_idx,
        position_ids,
        prefill_state.prefill_token_ranges if prefill_state is not None else None,
        prefill_state.swa_prefill_workspace_offsets if prefill_state is not None else None,
        image_left_arg,
        image_right_arg,
        swa_indices,
        swa_indices.stride(0),
        swa_lengths,
        swa_write_slots,
        swa_runtime_write_slots,
        token_to_batch_idx,
        b_seq_len,
        WRITE_SWA_RUNTIME=swa_runtime_write_slots is not None,
        SWA_PAGE_TOKEN_NUM=req_manager.cache_config.swa_page_token_num,
        SLIDING_WINDOW=sliding_window,
        WIDTH=width,
        HAS_IMAGE=image_left is not None,
        BLOCK_W=triton.next_power_of_2(width),
        SWA_CAPACITY_TOKENS=req_manager.swa_capacity_tokens,
        HOLD_REQUEST_ID=req_manager.HOLD_REQUEST_ID,
        HOLD_SWA_SLOT=prefill_state.HOLD_SWA_SLOT if prefill_state is not None else req_manager.HOLD_SWA_SLOT,
        IS_PREFILL=prefill_state is not None,
        num_warps=4,
    )
    return swa_indices, swa_lengths
