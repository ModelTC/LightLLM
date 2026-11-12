"""Sparse MLA attention directly on block16 NVFP4, with bounded split-K scratch.

K and V share the same compressed latent. Historical values are unpacked
inside each attention tile; no sequence-sized BF16 cache is materialized.
"""

import torch
import triton
import triton.language as tl

from ..kv_copy.nvfp4 import decode_e2m1


@triton.jit
def _prefill_queries(Req, CuQ, SeqLen, OutReq, OutLen, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start, end = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    req, length = tl.load(Req + batch), tl.load(SeqLen + batch)
    visible = length - (end - start) + offset + 1 if CAUSAL else length
    tl.store(OutReq + start + offset, req, offset < end - start)
    tl.store(OutLen + start + offset, visible, offset < end - start)


def prepare_prefill_queries(req_indices, cu_q, seq_lens, num_queries, max_q_len, causal):
    """Expand request IDs and causal lengths to one entry per prefill query."""
    query_req = torch.empty(num_queries, dtype=torch.int32, device=req_indices.device)
    lengths = torch.empty_like(query_req)
    _prefill_queries[(req_indices.numel(), triton.cdiv(max_q_len, 128))](
        req_indices, cu_q, seq_lens, query_req, lengths, causal, 128
    )
    return query_req, lengths


@triton.jit(do_not_specialize=["WIDTH", "SPLITS", "PER_SPLIT"])
def _attention(
    Q,
    QR,
    KV,
    Indices,
    Current,
    RowTable,
    QueryReq,
    Lengths,
    Partial,
    Lse,
    QS: tl.constexpr,
    QRS: tl.constexpr,
    KS: tl.constexpr,
    TS: tl.constexpr,
    CS: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROPE: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CURRENT: tl.constexpr,
    SPLITS,
    PER_SPLIT,
    BH: tl.constexpr,
    BK: tl.constexpr,
    DENSE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    h = tl.program_id(2) * BH + tl.arange(0, BH)
    d = tl.arange(0, BD)
    q = tl.load(Q + row * QS[0] + h[:, None] * QS[1] + d[None, :] * QS[2], (h[:, None] < HEADS) & (d[None, :] < DIM), 0)
    if ROPE:
        r = tl.arange(0, BR)
        qr = tl.load(
            QR + row * QRS[0] + h[:, None] * QRS[1] + r[None, :] * QRS[2],
            (h[:, None] < HEADS) & (r[None, :] < ROPE),
            0,
        )
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denom = tl.full((BH,), 0.0, tl.float32)
    acc = tl.full((BH, BD), 0.0, tl.float32)
    if DENSE:
        req = tl.load(QueryReq + row).to(tl.int64)
        length = tl.load(Lengths + row)
    for start in range(split * PER_SPLIT, (split + 1) * PER_SPLIT, BK):
        n = start + tl.arange(0, BK)
        if DENSE:
            index = tl.load(Indices + req * TS[0] + n * TS[1], (n < WIDTH) & (n < length), -1).to(tl.int64)
        else:
            index = tl.load(Indices + row * TS[0] + n * TS[1], n < WIDTH, -1).to(tl.int64)
        valid = (index >= 0) & (index < ROWS)
        base = KV + index[:, None] * KS
        packed = tl.load(base + d[None, :] // 2, valid[:, None] & (d[None, :] < DIM), 0)
        scale = (
            tl.load(base + DIM // 2 + d[None, :] // 16, valid[:, None] & (d[None, :] < DIM), 0)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        values = (decode_e2m1(packed, d[None, :]).to(tl.float32) * scale).to(Q.dtype.element_ty)
        if HAS_CURRENT:
            current_row = tl.load(RowTable + index, valid, -1).to(tl.int64)
            current = tl.load(
                Current + current_row[:, None] * CS[0] + d[None, :] * CS[2],
                valid[:, None] & (current_row[:, None] >= 0) & (d[None, :] < DIM),
                0,
            )
            values = tl.where(current_row[:, None] >= 0, current, values)
        logits = tl.dot(q, tl.trans(values))
        if ROPE:
            kr = tl.load(
                (base + DIM // 2 + DIM // 16).to(tl.pointer_type(Q.dtype.element_ty)) + r[None, :],
                valid[:, None] & (r[None, :] < ROPE),
                0,
            )
            if HAS_CURRENT:
                current_rope = tl.load(
                    Current + current_row[:, None] * CS[0] + (DIM + r[None, :]) * CS[2],
                    valid[:, None] & (current_row[:, None] >= 0) & (r[None, :] < ROPE),
                    0,
                )
                kr = tl.where(current_row[:, None] >= 0, current_rope, kr)
            logits += tl.dot(qr, tl.trans(kr))
        logits = tl.where(valid[None, :], logits * SCALE, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(logits, 1))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.exp(logits - safe_max[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(Q.dtype.element_ty), values)
        denom = denom * alpha + tl.sum(p, 1)
        maximum = new_max
    out_offset = ((row * SPLITS + split) * HEADS + h[:, None]) * DIM + d[None, :]
    tl.store(
        Partial + out_offset, acc / tl.where(denom > 0, denom, 1.0)[:, None], (h[:, None] < HEADS) & (d[None, :] < DIM)
    )
    tl.store(
        Lse + (row * SPLITS + split) * HEADS + h,
        tl.where(denom > 0, maximum + tl.log(denom), -float("inf")),
        h < HEADS,
    )


@triton.jit(do_not_specialize=["SPLITS"])
def _reduce(Partial, Lse, Out, HEADS: tl.constexpr, DIM: tl.constexpr, SPLITS, BS: tl.constexpr, BD: tl.constexpr):
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    s, d = tl.arange(0, BS), block * BD + tl.arange(0, BD)
    lse = tl.load(Lse + (row * SPLITS + s) * HEADS + head, s < SPLITS, -float("inf"))
    maximum = tl.max(lse, 0)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp(lse - safe)
    denom = tl.sum(weight, 0)
    partial = tl.load(
        Partial + ((row * SPLITS + s[:, None]) * HEADS + head) * DIM + d[None, :],
        (s[:, None] < SPLITS) & (d[None, :] < DIM),
        0,
    )
    value = tl.sum(partial * weight[:, None], 0) / tl.where(denom > 0, denom, 1.0)
    tl.store(Out + (row * HEADS + head) * DIM + d, value, d < DIM)


def nvfp4_mla_attention(
    q,
    packed_kv,
    indices,
    softmax_scale,
    q_rope=None,
    current_kv=None,
    row_table=None,
    query_req=None,
    lengths=None,
    max_seq_len=None,
):
    assert q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and q.ndim == 3
    assert packed_kv.dtype == torch.uint8 and packed_kv.ndim == 3 and packed_kv.shape[1] == 1
    assert packed_kv.is_cuda and packed_kv.device == q.device and packed_kv.stride(-1) == 1
    if indices.ndim == 3:
        assert indices.shape[1] == 1
        indices = indices[:, 0]
    dense = query_req is not None
    assert indices.ndim == 2 and (dense or indices.shape[0] == q.shape[0])
    assert indices.dtype in (torch.int32, torch.int64) and indices.device == q.device
    rows, heads, dim = q.shape
    rope = q_rope.shape[-1] if q_rope is not None else 0
    assert dim > 0 and dim % 32 == 0 and heads > 0
    assert packed_kv.shape[-1] == dim // 2 + dim // 16 + rope * 2
    if rope:
        assert q_rope.shape[:2] == q.shape[:2] and q_rope.dtype == q.dtype and q_rope.device == q.device
        assert rope % 16 == 0 and packed_kv.data_ptr() % 2 == packed_kv.stride(0) % 2 == 0
    if current_kv is not None:
        assert current_kv.ndim == 3 and current_kv.shape[1:] == (1, dim + rope)
        assert current_kv.dtype == q.dtype and current_kv.device == q.device
        assert row_table is not None and row_table.shape == (packed_kv.shape[0],)
        assert row_table.dtype in (torch.int32, torch.int64) and row_table.device == q.device
        assert row_table.is_contiguous()
    if dense:
        assert query_req.shape == lengths.shape == (rows,)
        assert query_req.dtype in (torch.int32, torch.int64) and lengths.dtype in (torch.int32, torch.int64)
        assert query_req.device == lengths.device == q.device
        assert query_req.is_contiguous() and lengths.is_contiguous()
        assert max_seq_len is not None and 0 <= max_seq_len <= indices.shape[1]
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if not rows:
        return output
    width = max_seq_len if dense else indices.shape[1]
    # Keep enough CTAs as the batch grows, without a small-batch cutoff that
    # would abruptly serialize K on a partially occupied GPU.
    head_tiles = triton.cdiv(heads, 16)
    desired_splits = max(1, min(triton.cdiv(width, 128), triton.cdiv(128, rows * head_tiles)))
    per_split = max(32, triton.cdiv(width, desired_splits * 32) * 32)
    splits = max(1, triton.cdiv(width, per_split))
    partial = torch.empty((rows, splits, heads, dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((rows, splits, heads), dtype=torch.float32, device=q.device)
    qr = q if q_rope is None else q_rope
    _attention[(rows, splits, triton.cdiv(heads, 16))](
        q,
        qr,
        packed_kv,
        indices,
        current_kv,
        row_table,
        query_req,
        lengths,
        partial,
        lse,
        q.stride(),
        qr.stride(),
        packed_kv.stride(0),
        indices.stride(),
        current_kv.stride() if current_kv is not None else (0, 0, 0),
        packed_kv.shape[0],
        width,
        heads,
        dim,
        rope,
        triton.next_power_of_2(dim),
        max(16, triton.next_power_of_2(rope)),
        float(softmax_scale),
        current_kv is not None,
        splits,
        per_split,
        16,
        32,
        dense,
        num_warps=8,
        num_stages=1,
    )
    _reduce[(rows, heads, triton.cdiv(dim, 128))](
        partial,
        lse,
        output,
        heads,
        dim,
        splits,
        triton.next_power_of_2(splits),
        128,
        num_warps=4,
    )
    return output
