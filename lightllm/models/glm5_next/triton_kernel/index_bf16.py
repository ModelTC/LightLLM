"""Unrotated BF16 Q/K with IEEE FP32 scoring for the indexer precision ablation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_mqa_logits(
    Q,
    K,
    Weights,
    Starts,
    Ends,
    Out,
    Q_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    POOLS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(1)
    pools = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, 32)
    start = tl.load(Starts + row).to(tl.int64)
    length = tl.load(Ends + row).to(tl.int64) - start
    valid = (pools < POOLS) & (pools < length)
    # One query per program preserves request boundaries in ragged/MTP batches.
    # IEEE FP32 avoids TF32 settings changing the precision experiment.
    dots = tl.full((BLOCK_H, BLOCK_P), 0, tl.float32)
    for block in range(tl.cdiv(DIM, 32)):
        cols = block * 32 + dims
        q = tl.load(
            Q + row * Q_STRIDE + heads[:, None] * Q_HEAD_STRIDE + cols[None, :],
            (heads[:, None] < HEADS) & (cols[None, :] < DIM),
            0,
        ).to(tl.float32)
        k = tl.load(
            K + (start + pools[None, :]) * DIM + cols[:, None],
            valid[None, :] & (cols[:, None] < DIM),
            0,
        ).to(tl.float32)
        dots = tl.dot(q, k, dots, input_precision="ieee")
    weights = tl.load(Weights + row * W_STRIDE + heads, heads < HEADS, 0).to(tl.float32)
    scores = tl.sum(tl.maximum(dots * (DIM**-0.5), 0) * weights[:, None], 0)
    tl.store(Out + row * POOLS + pools, tl.where(valid, scores, -float("inf")), pools < POOLS)


def bf16_mqa_logits(q, keys, weights, starts, ends, max_pools):
    """Return row-relative logits; weights already include heads**-0.5."""
    assert q.dtype == keys.dtype == torch.bfloat16
    assert keys.is_contiguous() and weights.stride(1) == 1
    logits = torch.empty((q.shape[0], max_pools), device=q.device, dtype=torch.float32)
    _bf16_mqa_logits[(triton.cdiv(max_pools, 32), q.shape[0])](
        q,
        keys,
        weights,
        starts,
        ends,
        logits,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        max_pools,
        q.shape[1],
        q.shape[2],
        max(16, triton.next_power_of_2(q.shape[1])),
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return logits
