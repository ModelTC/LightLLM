import torch
import triton
import triton.language as tl


@triton.jit
def _build_swa_index_kernel(
    req_idx_ptr,
    pos_ptr,
    req_to_token_ptr,
    req_to_token_stride0,
    full_to_swa_ptr,
    swa_index_ptr,
    swa_index_stride0,
    swa_length_ptr,
    WINDOW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req = tl.load(req_idx_ptr + token_idx).to(tl.int64)
    pos = tl.load(pos_ptr + token_idx).to(tl.int64)

    w = tl.arange(0, BLOCK_W)
    w_mask = w < WINDOW
    # most-recent-first window, identical to the eager _swa_indices (offset = position - arange).
    offset = pos - w
    valid = (offset >= 0) & w_mask
    safe_offset = tl.where(valid, offset, 0)
    full_slot = tl.load(req_to_token_ptr + req * req_to_token_stride0 + safe_offset, mask=valid, other=0).to(tl.int64)
    swa_slot = tl.load(full_to_swa_ptr + full_slot, mask=valid, other=-1)
    out = tl.where(valid, swa_slot, -1).to(tl.int32)
    tl.store(swa_index_ptr + token_idx * swa_index_stride0 + w, out, mask=w_mask)

    length = tl.minimum(tl.maximum(pos + 1, 1), WINDOW).to(tl.int32)
    tl.store(swa_length_ptr + token_idx, length)


def build_swa_index(
    req_idx: torch.Tensor,
    positions: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    full_to_swa_indexs: torch.Tensor,
    swa_index: torch.Tensor,
    swa_length: torch.Tensor,
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
    window = swa_index.shape[1]
    if T == 0:
        return swa_index, swa_length
    _build_swa_index_kernel[(T,)](
        req_idx,
        positions,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        full_to_swa_indexs,
        swa_index,
        swa_index.stride(0),
        swa_length,
        WINDOW=window,
        BLOCK_W=triton.next_power_of_2(window),
        num_warps=4,
    )
    return swa_index, swa_length
