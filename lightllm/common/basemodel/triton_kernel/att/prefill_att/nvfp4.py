"""Tensor Core prefill over NVFP4 KV, decoding E2M1 and E4M3 scales within each tile."""

import torch
import triton
import triton.language as tl

from ...kv_copy.nvfp4 import decode_e2m1


@triton.jit
def _prefill(
    Q,
    K,
    V,
    OUT,
    REQ,
    START,
    LENGTH,
    READY,
    TABLE,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    OS: tl.constexpr,
    TS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch, head = batch_head // HEADS, batch_head % HEADS
    kv_head = head // (HEADS // KV_HEADS)
    length = tl.load(LENGTH + batch)
    ready = tl.load(READY + batch)
    q_length = length - ready
    if tile * BM >= q_length:
        return
    start = tl.load(START + batch)
    req = tl.load(REQ + batch)
    m = tile * BM + tl.arange(0, BM)
    d = tl.arange(0, DIM)
    n = tl.arange(0, BN)
    q = tl.load(Q + (start + m[:, None]) * QS[0] + head * QS[1] + d[None, :] * QS[2], m[:, None] < q_length, other=0)
    sm_scale = 1.4426950408889634 / tl.sqrt(float(DIM))
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DIM), tl.float32)
    end = tl.minimum(length, ready + (tile + 1) * BM) if CAUSAL else length
    begin = tl.maximum(0, ready + tile * BM - LEFT) if LEFT >= 0 else 0
    for offset in range(begin, end, BN):
        positions = offset + n
        cache_valid = positions < end
        slots = tl.load(TABLE + req * TS[0] + positions * TS[1], cache_valid, other=0).to(tl.int64)
        kp = tl.load(
            K + slots[None, :] * KS[0] + kv_head * KS[1] + (d[:, None] // 2) * KS[2], cache_valid[None, :], other=0
        )
        k = decode_e2m1(kp, d[:, None]).to(tl.float32)
        sf = tl.load(
            K + slots[None, :] * KS[0] + kv_head * KS[1] + (DIM // 2 + d[:, None] // 16) * KS[2],
            cache_valid[None, :],
            other=0,
        )
        k = k * sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k = k.to(q.dtype)
        logits = tl.dot(q, k) * sm_scale
        valid = positions[None, :] < length
        if CAUSAL:
            valid = valid & (positions[None, :] <= (ready + m[:, None]))
        if LEFT >= 0:
            valid = valid & (positions[None, :] >= (ready + m[:, None] - LEFT))
        if RIGHT >= 0:
            valid = valid & (positions[None, :] <= (ready + m[:, None] + RIGHT))
        logits = tl.where(valid, logits, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(logits, 1))
        # Empty rows in an initial sliding-window tile must not introduce NaNs.
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp2(maximum - safe_max)
        p = tl.exp2(logits - safe_max[:, None])
        denominator = denominator * alpha + tl.sum(p, 1)
        vp = tl.load(
            V + slots[:, None] * VS[0] + kv_head * VS[1] + (d[None, :] // 2) * VS[2], cache_valid[:, None], other=0
        )
        v = decode_e2m1(vp, d[None, :]).to(tl.float32)
        sf = tl.load(
            V + slots[:, None] * VS[0] + kv_head * VS[1] + (DIM // 2 + d[None, :] // 16) * VS[2],
            cache_valid[:, None],
            other=0,
        )
        v = v * sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v = v.to(q.dtype)
        acc = tl.dot(p.to(q.dtype), v, acc * alpha[:, None])
        maximum = new_max
    result = acc * (1.0 / denominator[:, None])
    tl.store(OUT + (start + m[:, None]) * OS[0] + head * OS[1] + d[None, :] * OS[2], result, m[:, None] < q_length)


@torch.no_grad()
def context_attention_nvfp4(
    q,
    k,
    v,
    out,
    req,
    starts,
    lengths,
    ready,
    table,
    max_q_len,
    *,
    causal=True,
    sliding_window=(-1, -1),
):
    dim = q.shape[-1]
    assert dim in (16, 32, 64, 128, 256) and k.shape[-1] == dim // 2 + dim // 16
    assert q.shape[1] % k.shape[1] == 0 and q.dtype in (torch.float16, torch.bfloat16)
    bm = 64
    bn = 32 if dim >= 256 else 64
    num_warps = 4 if dim <= 128 else 8
    if max_q_len >= 128 and dim in (128, 256):
        bm = 128
        bn = 128 if dim == 128 else 32
        num_warps = 8
    _prefill[(triton.cdiv(max_q_len, bm), len(lengths) * q.shape[1])](
        q,
        k,
        v,
        out,
        req,
        starts,
        lengths,
        ready,
        table,
        q.stride(),
        k.stride(),
        v.stride(),
        out.stride(),
        table.stride(),
        q.shape[1],
        k.shape[1],
        dim,
        causal,
        sliding_window[0],
        sliding_window[1],
        bm,
        bn,
        num_warps=num_warps,
        num_stages=2,
    )
