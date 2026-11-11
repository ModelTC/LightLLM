"""Split-K Tensor Core decode over packed FP4, sharing KV tiles across MTP rows."""

import torch
import triton
import triton.language as tl

from ...kv_copy.nvfp4 import decode_e2m1
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


@triton.jit
def _decode(
    Q,
    K,
    V,
    TABLE,
    REQ,
    VISIBLE,
    POSITIONS,
    MARKERS,
    MID,
    LSE,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    TS: tl.constexpr,
    MS: tl.constexpr,
    LS: tl.constexpr,
    META_STRIDES: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    SHARED: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    BH: tl.constexpr,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    batch, kv_head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    size = tl.load(MARKERS + batch * META_STRIDES[3]) if SHARED else 1
    if size == 0:
        return
    lanes = tl.arange(0, ROWS)
    batch_lanes = lanes // BH
    head_lanes = lanes % BH
    valid = (batch_lanes < size) & (head_lanes < GROUP)
    batches = batch - size + 1 + tl.minimum(batch_lanes, size - 1)
    heads = kv_head * GROUP + tl.where(head_lanes < GROUP, head_lanes, 0)
    d = tl.arange(0, DIM)
    n = tl.arange(0, BN)
    q = tl.load(Q + batches[:, None] * QS[0] + heads[:, None] * QS[1] + d[None, :] * QS[2])
    visible = tl.load(VISIBLE + batches * META_STRIDES[1])
    positions = tl.load(POSITIONS + batches * META_STRIDES[2]) - 1
    maximum_length = tl.max(tl.where(batch_lanes < size, visible, 0), 0)
    req = tl.load(REQ + batch * META_STRIDES[0])
    sm_scale = 1.4426950408889634 / tl.sqrt(float(DIM))
    maximum = tl.full((ROWS,), -float("inf"), tl.float32)
    denominator = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, DIM), tl.float32)
    first_visible = tl.maximum(0, tl.min(tl.where(valid, positions, 0x7FFFFFFF), 0) - LEFT) if LEFT >= 0 else 0
    for block_start in range(first_visible + split * SPLIT, maximum_length, tl.num_programs(2) * SPLIT):
        end = tl.minimum(block_start + SPLIT, maximum_length)
        for offset in range(block_start, end, BN):
            tokens = offset + n
            slots = tl.load(TABLE + req * TS[0] + tokens * TS[1], tokens < end, other=0).to(tl.int64)
            kp = tl.load(
                K + slots[None, :] * KS[0] + kv_head * KS[1] + (d[:, None] // 2) * KS[2], tokens[None, :] < end, other=0
            )
            k = decode_e2m1(kp, d[:, None]).to(tl.float32)
            sf = tl.load(
                K + slots[None, :] * KS[0] + kv_head * KS[1] + (DIM // 2 + d[:, None] // 16) * KS[2],
                tokens[None, :] < end,
                other=0,
            )
            k = k * sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            k = k.to(q.dtype)
            logits = tl.dot(q, k) * sm_scale
            allowed = (tokens[None, :] < visible[:, None]) & (tokens[None, :] < end)
            if LEFT >= 0:
                allowed = allowed & (tokens[None, :] >= positions[:, None] - LEFT)
            if RIGHT >= 0:
                allowed = allowed & (tokens[None, :] <= positions[:, None] + RIGHT)
            logits = tl.where(allowed, logits, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(logits, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            p = tl.exp2(logits - safe_max[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            vp = tl.load(
                V + slots[:, None] * VS[0] + kv_head * VS[1] + (d[None, :] // 2) * VS[2], tokens[:, None] < end, other=0
            )
            v = decode_e2m1(vp, d[None, :]).to(tl.float32)
            sf = tl.load(
                V + slots[:, None] * VS[0] + kv_head * VS[1] + (DIM // 2 + d[None, :] // 16) * VS[2],
                tokens[:, None] < end,
                other=0,
            )
            v = v * sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            v = v.to(q.dtype)
            acc = tl.dot(p.to(q.dtype), v, acc * alpha[:, None])
            maximum = new_max
    result = acc * (1.0 / tl.where(denominator > 0, denominator, 1.0)[:, None])
    lse = tl.where(denominator > 0, maximum * 0.6931471805599453 + tl.log(denominator), -float("inf"))
    offsets = batches[:, None] * MS[0] + heads[:, None] * MS[1] + split * MS[2] + d[None, :] * MS[3]
    tl.store(MID + offsets, result, valid[:, None])
    tl.store(LSE + batches * LS[0] + heads * LS[1] + split * LS[2], lse, valid)


@triton.jit
def _reduce_splits(
    MID,
    LSE,
    OUT,
    MS: tl.constexpr,
    LS: tl.constexpr,
    OS: tl.constexpr,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    split = tl.arange(0, BLOCK)
    d = tl.arange(0, DIM)
    valid = split < SPLITS
    lse = tl.load(LSE + batch * LS[0] + head * LS[1] + split * LS[2], valid, other=-float("inf"))
    maximum = tl.max(lse, 0)
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(lse - safe_max)
    values = tl.load(
        MID + batch * MS[0] + head * MS[1] + split[:, None] * MS[2] + d[None, :] * MS[3], valid[:, None], other=0
    ).to(tl.float32)
    denominator = tl.sum(weights, 0)
    result = tl.sum(values * weights[:, None], 0) / tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + batch * OS[0] + head * OS[1] + d * OS[2], result)


@torch.no_grad()
def decode_attention_nvfp4(
    q,
    k,
    v,
    table,
    req,
    visible,
    original_lengths,
    max_kv_len,
    *,
    markers=None,
    max_group_size=1,
    sliding_window=(-1, -1),
    alloc_func=torch.empty,
):
    batch, heads, dim = q.shape
    assert dim in (16, 32, 64, 128, 256) and k.shape[-1] == dim // 2 + dim // 16
    assert heads % k.shape[1] == 0 and q.dtype in (torch.bfloat16, torch.float16)
    split = 256
    max_visible = min(max_kv_len, sliding_window[0] + max_group_size) if sliding_window[0] >= 0 else max_kv_len
    blocks = max(1, min(triton.cdiv(max_visible, split), 128 if batch <= 16 else (64 if batch <= 64 else 32)))
    mid = alloc_func((batch, heads, blocks, dim), dtype=q.dtype, device=q.device)
    lse = alloc_func((batch, heads, blocks), dtype=torch.float32, device=q.device)
    out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
    group = heads // k.shape[1]
    bh = triton.next_power_of_2(group)
    shared_rows = min(max_group_size, get_diverse_max_batch_shared_group_size()) if markers is not None else 1
    rows = max(16, bh * triton.next_power_of_2(shared_rows))
    _decode[(batch, k.shape[1], blocks)](
        q,
        k,
        v,
        table,
        req,
        visible,
        original_lengths,
        markers,
        mid,
        lse,
        q.stride(),
        k.stride(),
        v.stride(),
        table.stride(),
        mid.stride(),
        lse.stride(),
        (req.stride(0), visible.stride(0), original_lengths.stride(0), markers.stride(0) if markers is not None else 1),
        k.shape[1],
        group,
        dim,
        markers is not None,
        sliding_window[0],
        sliding_window[1],
        bh,
        rows,
        32 if dim == 256 else 64,
        split,
        num_warps=4 if rows <= 32 else 8,
        num_stages=2,
    )
    _reduce_splits[(batch, heads)](
        mid,
        lse,
        out,
        mid.stride(),
        lse.stride(),
        out.stride(),
        dim,
        blocks,
        triton.next_power_of_2(blocks),
        num_warps=4 if dim <= 128 else 8,
    )
    return out
