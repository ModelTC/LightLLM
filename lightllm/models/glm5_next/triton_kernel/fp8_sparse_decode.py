import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_sparse_direct_decode_kernel(
    q,
    packed_kv,
    topk,
    partial_o,
    partial_lse,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    packed_stride: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_d: tl.constexpr,
    packed_rows: tl.constexpr,
    softmax_scale: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLITS: tl.constexpr,
):
    qid = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    if HEADS <= BLOCK_H:
        h = tl.arange(0, BLOCK_H)
    else:
        h = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    h = h.to(tl.int64)
    d = tl.arange(0, 512).to(tl.int64)
    h_mask = h < HEADS
    n = tl.arange(0, BLOCK_K)
    slot = split * BLOCK_K + n
    indices = tl.load(topk + qid * topk_stride_b + slot * topk_stride_d, slot < 2176, -1).to(tl.int64)
    valid = (indices >= 0) & (indices < packed_rows)
    packed = packed_kv + indices[:, None] * packed_stride
    q_values = tl.load(
        q + qid * q_stride_b + h[:, None] * q_stride_h + d[None, :] * q_stride_d,
        h_mask[:, None],
        0.0,
    ).to(tl.bfloat16)
    group = tl.arange(0, 8)
    raw_scales = tl.load(
        packed + 576 + group[None, :],
        valid[:, None] & (group[None, :] < 7),
        127,
    )
    scales = (raw_scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    scales = tl.broadcast_to(scales[:, :, None], (BLOCK_K, 8, 64)).reshape(BLOCK_K, 512)
    fp8 = (
        tl.load(
            packed.to(tl.pointer_type(tl.float8e4nv)) + d[None, :],
            valid[:, None] & (d[None, :] < 448),
            0.0,
        ).to(tl.float32)
        * scales
    )
    tail = tl.load(
        (packed + 448).to(tl.pointer_type(tl.bfloat16)) + d[None, :] - 448,
        valid[:, None] & (d[None, :] >= 448),
        0.0,
    ).to(tl.float32)
    kv = tl.where(d[None, :] < 448, fp8, tail).to(tl.bfloat16)
    score = tl.dot(q_values, tl.trans(kv), out_dtype=tl.float32) * softmax_scale
    score = tl.where(valid[None, :], score, -float("inf"))
    maximum = tl.max(score, axis=1)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probability = tl.where(valid[None, :], tl.exp(score - safe_maximum[:, None]), 0.0)
    denominator = tl.sum(probability, axis=1)
    has_values = denominator > 0.0
    values = tl.dot(probability.to(tl.bfloat16), kv, out_dtype=tl.float32)
    base = (qid * SPLITS + split) * HEADS + h[:, None]
    tl.store(
        partial_o + base * 512 + d[None, :],
        tl.where(has_values[:, None], values / denominator[:, None], 0.0),
        h_mask[:, None],
    )
    tl.store(
        partial_lse + (qid * SPLITS + split) * HEADS + h,
        tl.where(has_values, maximum + tl.log(denominator), -float("inf")),
        h_mask,
    )


def _can_use_fp8_sparse_direct_decode(
    q_nope, packed_kv, topk, softmax_scale, kv_lora_rank, bound, is_cuda_graph_capture=False
):
    """Validate the GLM packed format and choose the direct decode implementation."""
    # The producer guarantees valid indices occupy at most the first 2051 slots.
    # The direct kernel scans 2112 slots; unknown bounds must retain the full native scan.
    if bound != 2051 or kv_lora_rank != 512 or q_nope.ndim != 3:
        return False
    batch, heads, dim = q_nope.shape
    if batch <= 0 or heads <= 0 or dim != 512:
        return False
    # Direct decode wins for small graph workloads; retain native eager/larger workloads.
    if not (is_cuda_graph_capture and batch * triton.cdiv(heads, 16) <= 8):
        return False
    if not isinstance(softmax_scale, (float, int)) or not math.isfinite(softmax_scale) or softmax_scale <= 0:
        return False
    return (
        q_nope.dtype == torch.bfloat16
        and q_nope.is_cuda
        and packed_kv.ndim == 3
        and packed_kv.shape[1] == 1
        and packed_kv.shape[2] >= 584
        and packed_kv.dtype == torch.uint8
        and packed_kv.device == q_nope.device
        and packed_kv.stride(-1) == 1
        # The tail is loaded through a BF16 pointer, requiring two-byte alignment.
        and packed_kv.data_ptr() % 2 == 0
        and packed_kv.stride(0) % 2 == 0
        and topk.shape == (batch, 1, 2176)
        and topk.dtype in (torch.int32, torch.int64)
        and topk.device == q_nope.device
    )


def _fp8_sparse_direct_decode(q_nope, packed_kv, topk, softmax_scale):
    from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
        _reduce_direct_smallq_kernel,
    )

    batch = q_nope.shape[0]
    heads = q_nope.shape[1]
    block_h = 16 if batch * heads <= 64 else min(32, max(16, triton.next_power_of_2(heads)))
    block_k = 16 if batch == 1 and heads <= 16 else 64
    splits = 2112 // block_k
    partial_o = torch.empty((batch, splits, heads, 512), device=q_nope.device, dtype=torch.float32)
    partial_lse = torch.empty((batch, splits, heads), device=q_nope.device, dtype=torch.float32)
    out = torch.empty(q_nope.shape, device=q_nope.device, dtype=q_nope.dtype)
    launch_grid = (batch, splits) if heads <= block_h else (batch, splits, triton.cdiv(heads, block_h))
    _fp8_sparse_direct_decode_kernel[launch_grid](
        q_nope,
        packed_kv,
        topk,
        partial_o,
        partial_lse,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        packed_kv.stride(0),
        topk.stride(0),
        topk.stride(2),
        packed_kv.shape[0],
        float(softmax_scale),
        heads,
        block_h,
        block_k,
        splits,
        num_warps=4 if block_k == 16 else 8,
        num_stages=1,
    )
    _reduce_direct_smallq_kernel[(batch, heads, 4)](
        partial_o,
        partial_lse,
        out,
        heads,
        splits,
        triton.next_power_of_2(splits),
        num_warps=4,
    )
    return out
