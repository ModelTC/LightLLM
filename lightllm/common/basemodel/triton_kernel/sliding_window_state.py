import math
import torch
import triton
import triton.language as tl


@triton.jit
def _build_sliding_window_page_table(
    PageTable,
    BKVStart,
    BReqIdx,
    BSeqLen,
    BReadyCacheLen,
    BQStartLoc,
    MemIndexes,
    table_width,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req_idx = tl.load(BReqIdx + batch)
    history = tl.load(BReadyCacheLen + batch)
    seq_len = tl.load(BSeqLen + batch)
    q_start = tl.load(BQStartLoc + batch)
    kv_start = tl.maximum(history - WINDOW, 0)
    positions = kv_start + offsets
    is_new = positions >= history
    new_index = tl.load(MemIndexes + q_start + positions - history, mask=is_new & (positions < seq_len), other=0)
    index = tl.where(is_new, new_index, req_idx * WINDOW + positions % WINDOW)
    tl.store(
        PageTable + batch * table_width + offsets, tl.where(positions < seq_len, index, -1), mask=offsets < table_width
    )
    if tl.program_id(1) == 0:
        tl.store(BKVStart + batch, kv_start)


@torch.no_grad()
def build_sliding_window_page_table(
    b_req_idx, b_seq_len, b_ready_cache_len, b_q_start_loc, mem_indexes, window, max_q_seq_len
):
    """Batch-local, chronological table over fixed ring history and this prefill's new KV.

    Column zero represents b_kv_start_pos, not absolute token zero. No KV is moved.
    """
    page_table = torch.empty((b_req_idx.numel(), window + max_q_seq_len), dtype=torch.int32, device=b_req_idx.device)
    b_kv_start_pos = torch.empty_like(b_req_idx)
    _build_sliding_window_page_table[(b_req_idx.numel(), triton.cdiv(page_table.shape[1], 256))](
        page_table,
        b_kv_start_pos,
        b_req_idx,
        b_seq_len,
        b_ready_cache_len,
        b_q_start_loc,
        mem_indexes,
        page_table.shape[1],
        WINDOW=window,
        BLOCK=256,
        num_warps=4,
    )
    return page_table, b_kv_start_pos


@triton.jit
def _commit_sliding_window_kv(
    Pool,
    MemIndexes,
    BReqIdx,
    BSeqLen,
    BReadyCacheLen,
    BQStartLoc,
    stride_layer,
    stride_token,
    WINDOW: tl.constexpr,
    KV_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, layer = tl.program_id(0), tl.program_id(1)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    req_idx = tl.load(BReqIdx + batch).to(tl.int64)
    history = tl.load(BReadyCacheLen + batch)
    seq_len = tl.load(BSeqLen + batch)
    q_start = tl.load(BQStartLoc + batch)
    positions = seq_len - WINDOW + offsets // KV_DIM
    # Retained old history is already in place. Copy only this chunk's newest W KV.
    mask = (offsets < WINDOW * KV_DIM) & (positions >= history)
    src_index = tl.load(MemIndexes + q_start + positions - history, mask=mask, other=0).to(tl.int64)
    dst_index = req_idx * WINDOW + positions % WINDOW
    layer_ptr = Pool + layer.to(tl.int64) * stride_layer
    values = tl.load(layer_ptr + src_index * stride_token + offsets % KV_DIM, mask=mask, other=0)
    tl.store(layer_ptr + dst_index * stride_token + offsets % KV_DIM, values, mask=mask)


@torch.no_grad()
def commit_sliding_window_kv(pool, mem_indexes, b_req_idx, b_seq_len, b_ready_cache_len, b_q_start_loc, window):
    """Commit [layer, slot, ...payload] tails after every prefill reader has finished.

    The contiguous per-token payload can be KV heads, an MLA vector or packed bytes.
    """
    kv_dim = math.prod(pool.shape[2:])
    _commit_sliding_window_kv[(b_req_idx.numel(), pool.shape[0], triton.cdiv(window * kv_dim, 1024))](
        pool,
        mem_indexes,
        b_req_idx,
        b_seq_len,
        b_ready_cache_len,
        b_q_start_loc,
        pool.stride(0),
        pool.stride(1),
        WINDOW=window,
        KV_DIM=kv_dim,
        BLOCK=1024,
        num_warps=4,
    )


@triton.jit
def _get_sliding_window_decode_indexes(Out, BReqIdx, BSeqLen, WINDOW: tl.constexpr):
    batch = tl.program_id(0)
    req_idx = tl.load(BReqIdx + batch)
    seq_len = tl.load(BSeqLen + batch)
    tl.store(Out + batch, req_idx * WINDOW + (seq_len - 1) % WINDOW)


@torch.no_grad()
def get_sliding_window_decode_indexes(b_req_idx, b_seq_len, window):
    """Single-token decode writes directly into each request's fixed ring."""
    indexes = torch.empty_like(b_req_idx)
    _get_sliding_window_decode_indexes[(b_req_idx.numel(),)](indexes, b_req_idx, b_seq_len, WINDOW=window, num_warps=1)
    return indexes
