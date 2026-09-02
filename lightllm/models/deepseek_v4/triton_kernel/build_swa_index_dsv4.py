import torch
import triton
import triton.language as tl


@triton.jit
def _build_image_visibility_kernel(
    image_spans_ptr,  # (num_images, 3): batch index, absolute image start, visible image length
    b_q_start_loc_ptr,
    b_ready_cache_len_ptr,
    b_q_seq_len_ptr,
    image_left_ptr,
    image_right_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    image_idx = tl.program_id(0)
    span_ptr = image_spans_ptr + image_idx * 3
    batch_idx = tl.load(span_ptr)
    image_start = tl.load(span_ptr + 1)
    image_len = tl.load(span_ptr + 2)
    cache_len = tl.load(b_ready_cache_len_ptr + batch_idx)
    q_seq_len = tl.load(b_q_seq_len_ptr + batch_idx)
    flat_image_start = tl.load(b_q_start_loc_ptr + batch_idx) + image_start - cache_len

    for start in range(0, image_len, BLOCK_SIZE):
        offset = start + tl.arange(0, BLOCK_SIZE)
        q_offset = image_start - cache_len + offset
        mask = (offset < image_len) & (q_offset >= 0) & (q_offset < q_seq_len)
        output_offset = flat_image_start + offset
        tl.store(image_left_ptr + output_offset, offset, mask=mask)
        tl.store(image_right_ptr + output_offset, image_len - 1 - offset, mask=mask)


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
def _build_swa_index_kernel(
    req_idx_ptr,
    pos_ptr,
    req_to_token_ptr,
    req_to_token_stride0,
    full_to_swa_ptr,
    image_left_ptr,
    image_right_ptr,
    swa_index_ptr,
    swa_index_stride0,
    swa_length_ptr,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_IMAGE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req = tl.load(req_idx_ptr + token_idx).to(tl.int64)
    pos = tl.load(pos_ptr + token_idx).to(tl.int64)

    w = tl.arange(0, BLOCK_W)
    w_mask = w < WIDTH
    offset = pos - w
    valid = (offset >= 0) & (w < WINDOW)
    length = tl.minimum(tl.maximum(pos + 1, 1), WINDOW)
    if HAS_IMAGE:
        left = tl.load(image_left_ptr + token_idx)
        right = tl.load(image_right_ptr + token_idx)
        is_image = (left > 0) | (right > 0)
        image_start = tl.maximum(pos - (WINDOW - 1) - tl.maximum(left - (WINDOW - 1), 0), 0)
        image_length = pos + right - image_start + 1
        offset = tl.where(is_image, image_start + w, offset)
        valid = tl.where(is_image, w < image_length, valid)
        length = tl.where(is_image, image_length, length)
    safe_offset = tl.where(valid, offset, 0)
    full_slot = tl.load(req_to_token_ptr + req * req_to_token_stride0 + safe_offset, mask=valid, other=0).to(tl.int64)
    swa_slot = tl.load(full_to_swa_ptr + full_slot, mask=valid, other=-1)
    out = tl.where(valid, swa_slot, -1).to(tl.int32)
    tl.store(swa_index_ptr + token_idx * swa_index_stride0 + w, out, mask=w_mask)

    tl.store(swa_length_ptr + token_idx, length.to(tl.int32))


def build_swa_index(
    req_idx: torch.Tensor,
    positions: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    full_to_swa_indexs: torch.Tensor,
    swa_index: torch.Tensor,
    swa_length: torch.Tensor,
    window: int = None,
    image_left: torch.Tensor = None,
    image_right: torch.Tensor = None,
):
    """Per-token sliding-window FlashMLA index table, built ONCE per forward (layer-independent:
    full_to_swa is a single global map and the window is a model constant, so every layer's swa
    indices are identical). Replaces DeepseekV4IndexInfer._swa_indices: for token t at
    (req_idx, position) gather the last `window` tokens' full slots via req_to_token, then map
    full -> swa; out-of-range positions store -1.

    Writes (swa_index [T, window] int32, swa_length [T] int32). The caller owns the output storage;
    the reader adds the s_q axis via unsqueeze(1).
    """
    T = positions.shape[0]
    width = swa_index.shape[1]
    window = width if window is None else int(window)
    assert window <= width
    assert (image_left is None) == (image_right is None)
    if T == 0:
        return swa_index, swa_length
    image_left_arg = positions if image_left is None else image_left
    image_right_arg = positions if image_right is None else image_right
    _build_swa_index_kernel[(T,)](
        req_idx,
        positions,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        full_to_swa_indexs,
        image_left_arg,
        image_right_arg,
        swa_index,
        swa_index.stride(0),
        swa_length,
        WINDOW=window,
        WIDTH=width,
        HAS_IMAGE=image_left is not None,
        BLOCK_W=triton.next_power_of_2(width),
        num_warps=4,
    )
    return swa_index, swa_length
