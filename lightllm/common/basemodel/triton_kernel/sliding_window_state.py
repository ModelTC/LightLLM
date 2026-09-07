import torch
import triton
import triton.language as tl


@triton.jit
def _move_sliding_window(
    Pool,
    BReqIdx,
    BSeqLen,
    BReadyCacheLen,
    BQStartLoc,
    runtime_token_start,
    stride_layer,
    stride_token,
    WINDOW: tl.constexpr,
    KV_DIM: tl.constexpr,
    COMPACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    layer_idx = tl.program_id(1).to(tl.int64)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    req_idx = tl.load(BReqIdx + batch_idx).to(tl.int64)
    history_len = tl.load(BReadyCacheLen + batch_idx).to(tl.int64)
    q_start = tl.load(BQStartLoc + batch_idx).to(tl.int64)
    current_start = tl.cast(runtime_token_start, tl.int64) + q_start + (batch_idx + 1) * WINDOW
    if COMPACT:
        end = tl.load(BSeqLen + batch_idx).to(tl.int64)
    else:
        end = history_len
    positions = end - WINDOW + offsets // KV_DIM
    canonical = req_idx * WINDOW + positions % WINDOW
    active = current_start + positions - history_len
    if COMPACT:
        src, dst = active, canonical
    else:
        src, dst = canonical, active
    layer_offset = layer_idx * tl.cast(stride_layer, tl.int64)
    mask = (offsets < WINDOW * KV_DIM) & (positions >= 0)
    values = tl.load(Pool + layer_offset + src * stride_token + offsets % KV_DIM, mask=mask, other=0)
    tl.store(Pool + layer_offset + dst * stride_token + offsets % KV_DIM, values, mask=mask)


@torch.no_grad()
def move_sliding_window(
    pool,
    b_req_idx,
    b_seq_len,
    b_ready_cache_len,
    b_q_start_loc,
    window,
    runtime_token_start,
    compact=False,
):
    """Move every owner's window between canonical rings and the prefill region."""
    assert pool.ndim == 4 and pool.stride(-1) == 1 and pool.stride(-2) == pool.shape[-1]
    if not b_req_idx.numel():
        return
    kv_dim = pool.shape[2] * pool.shape[3]
    grid = (b_req_idx.numel(), pool.shape[0], triton.cdiv(window * kv_dim, 1024))
    _move_sliding_window[grid](
        pool,
        b_req_idx,
        b_seq_len,
        b_ready_cache_len,
        b_q_start_loc,
        runtime_token_start,
        pool.stride(0),
        pool.stride(1),
        WINDOW=window,
        KV_DIM=kv_dim,
        COMPACT=compact,
        BLOCK=1024,
        num_warps=4,
    )


@triton.jit
def _get_sliding_window_mem_indexes(
    Out,
    BReqIdx,
    BSeqLen,
    BQSeqLen,
    BQStartLoc,
    runtime_token_start,
    WINDOW: tl.constexpr,
    IS_PREFILL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if IS_PREFILL:
        q_start = tl.load(BQStartLoc + batch_idx).to(tl.int64)
        q_len = tl.load(BQSeqLen + batch_idx)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        current_start = tl.cast(runtime_token_start, tl.int64) + q_start + (batch_idx + 1) * WINDOW
        tl.store(Out + q_start + offsets, current_start + offsets, mask=offsets < q_len)
    else:
        req_idx = tl.load(BReqIdx + batch_idx).to(tl.int64)
        seq_len = tl.load(BSeqLen + batch_idx).to(tl.int64)
        tl.store(Out + batch_idx, req_idx * WINDOW + (seq_len - 1) % WINDOW)


@torch.no_grad()
def get_sliding_window_mem_indexes(
    b_req_idx,
    b_seq_len,
    b_q_seq_len,
    b_q_start_loc,
    window,
    runtime_token_start,
    total_token_num,
    max_q_seq_len,
    is_prefill,
):
    indexes = torch.empty(total_token_num, dtype=torch.int32, device=b_req_idx.device)
    if not b_req_idx.numel():
        return indexes
    grid = (b_req_idx.numel(), triton.cdiv(max_q_seq_len, 256) if is_prefill else 1)
    _get_sliding_window_mem_indexes[grid](
        indexes,
        b_req_idx,
        b_seq_len,
        b_q_seq_len,
        b_q_start_loc,
        runtime_token_start,
        WINDOW=window,
        IS_PREFILL=is_prefill,
        BLOCK=256,
        num_warps=4 if is_prefill else 1,
    )
    return indexes
