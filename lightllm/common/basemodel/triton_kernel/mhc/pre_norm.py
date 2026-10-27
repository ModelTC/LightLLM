# SPDX-License-Identifier: Apache-2.0

"""mHC pre-mixing fused with the following RMSNorm."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from lightllm.common.triton_utils.autotuner import autotune

from .configs import (
    decode_post_pre_configs,
    decode_post_pre_key,
    pre_norm_configs,
    pre_norm_key,
    projection_configs,
    projection_key,
)
from .post import _hc_post_values, hc_post

LIGHTLLM_DISABLE_DEEPGEMM_MHC = os.getenv("LIGHTLLM_DISABLE_DEEPGEMM_MHC", "False").upper() in ["ON", "TRUE", "1"]


@triton.jit
def _hc_mix_weights(
    gemm_partial,  # 输入 [P, T, M]，fp32；投影的 split-K 部分和。
    sqrsum_partial,  # 输入 [P, T]，fp32；残差平方和的 split-K 部分和。
    scale,  # 输入 [3]，fp32；pre、post、residual 三组缩放系数。
    base,  # 输入 [M]，fp32；M = 2 * S + S * S。
    token,
    gemm_stride_s,
    gemm_stride_m: tl.constexpr,
    sqrsum_stride_s,
    sqrsum_stride_m: tl.constexpr,
    FLATTENED_HIDDEN: tl.constexpr,
    RMS_EPS: tl.constexpr,
    STREAMS: tl.constexpr,
    HC_EPS: tl.constexpr,
    POST_MULTIPLIER: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    N_SPLITS,
    BLOCK_SPLITS: tl.constexpr,
    UNROLL_SINKHORN: tl.constexpr,
):
    """Reduce split-K partitions in parallel and compute the three mixing weights."""
    splits = tl.arange(0, BLOCK_SPLITS)
    stream_offsets = tl.arange(0, STREAMS)
    matrix_offsets = tl.arange(0, STREAMS * STREAMS)
    partial_base = gemm_partial + splits[:, None] * gemm_stride_s + token * gemm_stride_m
    pre_raw = tl.sum(tl.load(partial_base + stream_offsets[None, :], splits[:, None] < N_SPLITS, 0.0), axis=0)
    post_raw = tl.sum(
        tl.load(partial_base + STREAMS + stream_offsets[None, :], splits[:, None] < N_SPLITS, 0.0), axis=0
    )
    matrix_raw = tl.sum(
        tl.load(partial_base + 2 * STREAMS + matrix_offsets[None, :], splits[:, None] < N_SPLITS, 0.0), axis=0
    )
    sqrsum = tl.sum(
        tl.load(sqrsum_partial + splits * sqrsum_stride_s + token * sqrsum_stride_m, splits < N_SPLITS, 0.0),
        axis=0,
    )
    inv_rms = tl.rsqrt(sqrsum / FLATTENED_HIDDEN + RMS_EPS)
    pre_raw *= inv_rms
    post_raw *= inv_rms
    matrix_raw *= inv_rms

    pre_values = tl.sigmoid(pre_raw * tl.load(scale) + tl.load(base + stream_offsets)) + HC_EPS
    post_values = POST_MULTIPLIER * tl.sigmoid(post_raw * tl.load(scale + 1) + tl.load(base + STREAMS + stream_offsets))

    residual_logits = matrix_raw * tl.load(scale + 2) + tl.load(base + 2 * STREAMS + matrix_offsets)
    residual_logits = tl.reshape(residual_logits, (STREAMS, STREAMS))
    residual_logits = residual_logits - tl.max(residual_logits, axis=1)[:, None]
    matrix = tl.exp(residual_logits)
    # Keep the original division form instead of explicitly approximating reciprocals.
    matrix = matrix / tl.sum(matrix, axis=1)[:, None]
    matrix += HC_EPS
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)
    for _ in tl.range(1, SINKHORN_ITERS, loop_unroll_factor=max(1, SINKHORN_ITERS - 1) if UNROLL_SINKHORN else 1):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + HC_EPS)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)

    return pre_values, post_values, matrix


@triton.jit(do_not_specialize=["N_SPLITS"])
def _hc_pre_norm_kernel(
    x,  # 输入 [T, S * H]，bf16；逻辑形状 [T, S, H]。
    gemm_partial,
    sqrsum_partial,
    scale,
    base,
    post,
    residual_mix,
    norm_weight,  # 输入 [H]，子层 RMSNorm 权重，通常为 bf16。
    output,  # 输出 [T, H]，bf16；加权合并后再做 RMSNorm。
    hidden: tl.constexpr,
    x_stride_m: tl.constexpr,
    gemm_stride_s,
    gemm_stride_m: tl.constexpr,
    sqrsum_stride_s,
    sqrsum_stride_m: tl.constexpr,
    out_stride_m: tl.constexpr,
    STREAMS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    POST_MULTIPLIER: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    N_SPLITS,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    UNROLL_SINKHORN: tl.constexpr = True,
):
    """Fuse split reduction, Sinkhorn, stream mixing and sublayer RMSNorm."""
    token = tl.program_id(0)
    pre_values, post_values, matrix = _hc_mix_weights(
        gemm_partial,
        sqrsum_partial,
        scale,
        base,
        token,
        gemm_stride_s,
        gemm_stride_m,
        sqrsum_stride_s,
        sqrsum_stride_m,
        STREAMS * hidden,
        RMS_EPS,
        STREAMS,
        HC_EPS,
        POST_MULTIPLIER,
        SINKHORN_ITERS,
        N_SPLITS,
        BLOCK_SPLITS,
        UNROLL_SINKHORN,
    )
    tl.store(post + token * STREAMS + tl.arange(0, STREAMS), post_values)
    tl.store(
        residual_mix + token * STREAMS * STREAMS + tl.arange(0, STREAMS * STREAMS),
        tl.reshape(matrix, (STREAMS * STREAMS,)),
    )
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for stream in tl.static_range(STREAMS):
        residual = tl.load(
            x + token * x_stride_m + stream * hidden + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        pre_value = tl.sum(tl.where(tl.arange(0, STREAMS) == stream, pre_values, 0.0), axis=0)
        accumulator += residual * pre_value

    # Preserve the checkpoint's bf16 rounding between stream mixing and RMSNorm.
    rounded = accumulator.to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(rounded * rounded, axis=0) / hidden
    inv_rms = tl.rsqrt(variance + NORM_EPS)
    weight = tl.load(norm_weight + hidden_offsets, mask=hidden_mask, other=0.0)
    tl.store(
        output + token * out_stride_m + hidden_offsets,
        rounded * inv_rms * weight,
        mask=hidden_mask,
    )


def _compute_prenorm_splits(tokens: int, flattened_hidden: int, device: torch.device) -> int:
    """由 T = tokens、K = flattened_hidden 和设备 SM 数确定 split-K 分片数 P。

    tokens 和 flattened_hidden 为整数，device 指定 CUDA 设备；返回整数 P。
    P 决定 gemm_partial [P, T, M] 和 sqrsum_partial [P, T] 的首维，
    其中 M = 2 * S + S * S，S 为 stream 数。
    """
    grid_size = triton.cdiv(tokens, 64)
    k_blocks = triton.cdiv(flattened_hidden, 64)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return max(1, min(sms // max(grid_size, 1), k_blocks // 4))


@triton.jit
def _hc_post_prenorm_kernel(
    layer_output,
    residual,
    residual_mix,
    post_mix,
    fn,
    updated_residual,
    gemm_partial,
    sqrsum_partial,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SPLIT_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Fuse post-mixing and the next projection for small decode batches."""
    token = tl.program_id(0)
    mix_tile = tl.program_id(1)
    split = tl.program_id(2)
    columns = split * SPLIT_H + tl.arange(0, BLOCK_H)
    mask = (columns < HIDDEN) & (tl.arange(0, BLOCK_H) < SPLIT_H)
    values = tl.load(layer_output + token * HIDDEN + columns, mask, 0.0).to(tl.float32)
    residual_ptr = residual + token * 4 * HIDDEN + columns
    r0 = tl.load(residual_ptr, mask, 0.0).to(tl.float32)
    r1 = tl.load(residual_ptr + HIDDEN, mask, 0.0).to(tl.float32)
    r2 = tl.load(residual_ptr + 2 * HIDDEN, mask, 0.0).to(tl.float32)
    r3 = tl.load(residual_ptr + 3 * HIDDEN, mask, 0.0).to(tl.float32)
    r0, r1, r2, r3 = _hc_post_values(values, r0, r1, r2, r3, residual_mix + token * 16, post_mix + token * 4)
    # The following pre must see the same bf16 residual as a standalone post.
    r0 = r0.to(tl.bfloat16).to(tl.float32)
    r1 = r1.to(tl.bfloat16).to(tl.float32)
    r2 = r2.to(tl.bfloat16).to(tl.float32)
    r3 = r3.to(tl.bfloat16).to(tl.float32)
    if mix_tile == 0:
        updated_ptr = updated_residual + token * 4 * HIDDEN + columns
        tl.store(updated_ptr, r0, mask)
        tl.store(updated_ptr + HIDDEN, r1, mask)
        tl.store(updated_ptr + 2 * HIDDEN, r2, mask)
        tl.store(updated_ptr + 3 * HIDDEN, r3, mask)
        sqrsum = tl.sum(tl.where(mask, r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3, 0.0), 0)
        tl.store(sqrsum_partial + split * TOKENS + token, sqrsum)
    mixes = mix_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    weights_ptr = fn + mixes[:, None] * 4 * HIDDEN + columns[None, :]
    weights_mask = (mixes[:, None] < 24) & mask[None, :]
    products = tl.load(weights_ptr, weights_mask, 0.0) * r0[None, :]
    products += tl.load(weights_ptr + HIDDEN, weights_mask, 0.0) * r1[None, :]
    products += tl.load(weights_ptr + 2 * HIDDEN, weights_mask, 0.0) * r2[None, :]
    products += tl.load(weights_ptr + 3 * HIDDEN, weights_mask, 0.0) * r3[None, :]
    projections = tl.sum(products, 1)
    tl.store(gemm_partial + split * TOKENS * 24 + token * 24 + mixes, projections, mixes < 24)


@autotune(
    kernel_name="mhc_pre_norm:v1",
    configs_gen_func=pre_norm_configs,
    static_key_func=pre_norm_key,
    run_key_func=lambda x: x.shape[0],
    benchmark_repeats=32,
    benchmark_retries=5,
)
def _finish_pre_norm(
    x,
    gemm_partial,
    sqrsum_partial,
    scale,
    base,
    norm_weight,
    streams,
    rms_eps,
    norm_eps,
    hc_eps,
    sinkhorn_iters,
    post_multiplier,
    run_config=None,
):
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    n_splits = gemm_partial.shape[0]
    post = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
    residual_mix = torch.empty((tokens, streams, streams), dtype=torch.float32, device=x.device)
    layer_input = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    _hc_pre_norm_kernel[(tokens,)](
        x,
        gemm_partial,
        sqrsum_partial,
        scale,
        base,
        post,
        residual_mix,
        norm_weight,
        layer_input,
        hidden=hidden,
        x_stride_m=x.stride(0),
        gemm_stride_s=gemm_partial.stride(0),
        gemm_stride_m=gemm_partial.stride(1),
        sqrsum_stride_s=sqrsum_partial.stride(0),
        sqrsum_stride_m=sqrsum_partial.stride(1),
        out_stride_m=layer_input.stride(0),
        RMS_EPS=rms_eps,
        STREAMS=streams,
        HC_EPS=hc_eps,
        POST_MULTIPLIER=post_multiplier,
        SINKHORN_ITERS=sinkhorn_iters,
        N_SPLITS=n_splits,
        NORM_EPS=norm_eps,
        BLOCK_SPLITS=triton.next_power_of_2(n_splits),
        BLOCK_H=triton.next_power_of_2(hidden),
        UNROLL_SINKHORN=run_config.get("unroll_sinkhorn", True) if run_config else True,
        num_warps=run_config["num_warps"] if run_config else 4,
    )
    return layer_input, residual_mix, post


@autotune(
    kernel_name="mhc_prenorm_projection:v1",
    configs_gen_func=projection_configs,
    static_key_func=projection_key,
    run_key_func=lambda x: x.shape[0],
    benchmark_repeats=32,
    benchmark_retries=5,
)
def _hc_prenorm_projection(x, fn, run_config=None):
    from deep_gemm import tf32_hc_prenorm_gemm

    tokens, flattened_hidden = x.shape
    n_splits = run_config.get("n_splits", 0) if run_config else 0
    if not n_splits:
        n_splits = _compute_prenorm_splits(tokens, flattened_hidden, x.device)
    n_splits = max(1, min(n_splits, triton.cdiv(flattened_hidden, 64) // 4))
    gemm_partial = torch.empty((n_splits, tokens, fn.shape[0]), dtype=torch.float32, device=x.device)
    sqrsum_partial = torch.empty((n_splits, tokens), dtype=torch.float32, device=x.device)
    tf32_hc_prenorm_gemm(x, fn, gemm_partial, sqrsum_partial, n_splits)
    return gemm_partial, sqrsum_partial


def _hc_post_prenorm_projection(layer_output, residual, residual_mix, post_mix, fn, run_config=None):
    tokens, hidden = layer_output.shape
    if run_config is None:
        run_config = {"n_splits": 8, "BLOCK_M": 1 if tokens < 8 else 2 if tokens < 16 else 4, "num_warps": 4}
    n_splits = run_config["n_splits"]
    block_m = run_config["BLOCK_M"]
    split_h = triton.cdiv(hidden, n_splits)
    updated = torch.empty_like(residual)
    gemm_partial = torch.empty((n_splits, tokens, 24), dtype=torch.float32, device=residual.device)
    sqrsum_partial = torch.empty((n_splits, tokens), dtype=torch.float32, device=residual.device)
    _hc_post_prenorm_kernel[(tokens, triton.cdiv(24, block_m), n_splits)](
        layer_output,
        residual,
        residual_mix,
        post_mix,
        fn,
        updated,
        gemm_partial,
        sqrsum_partial,
        TOKENS=tokens,
        HIDDEN=hidden,
        SPLIT_H=split_h,
        BLOCK_H=triton.next_power_of_2(split_h),
        BLOCK_M=block_m,
        num_warps=run_config["num_warps"],
    )
    return updated, gemm_partial, sqrsum_partial


@autotune(
    kernel_name="mhc_post_pre_norm_decode:v1",
    configs_gen_func=decode_post_pre_configs,
    static_key_func=decode_post_pre_key,
    run_key_func=lambda layer_output: layer_output.shape[0],
    benchmark_repeats=32,
    benchmark_retries=5,
)
def _hc_post_pre_norm_decode(
    layer_output,
    residual,
    residual_mix,
    post_mix,
    fn,
    scale,
    base,
    norm_weight,
    streams,
    rms_eps,
    norm_eps,
    hc_eps,
    sinkhorn_iters,
    post_multiplier,
    run_config=None,
):
    updated, gp, sp = _hc_post_prenorm_projection(
        layer_output, residual, residual_mix, post_mix, fn, run_config=run_config
    )
    norm_config = {"num_warps": run_config["norm_num_warps"] if run_config else 4, "unroll_sinkhorn": True}
    x, mix, post = _finish_pre_norm(
        updated,
        gp,
        sp,
        scale,
        base,
        norm_weight,
        streams,
        rms_eps,
        norm_eps,
        hc_eps,
        sinkhorn_iters,
        post_multiplier,
        run_config=norm_config,
    )
    return x, updated, mix, post


def hc_pre_norm(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    streams: int,
    rms_eps: float,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_multiplier: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 mHC pre-mixing 与紧随其后的子层 RMSNorm 融合。

    T = tokens，S = streams，H = 单条 stream 的 hidden size，M = 2 * S + S * S。

    Args:
        x: [T, S * H]，连续的 bf16 残差激活；逻辑形状 [T, S, H]。
        fn: [M, S * H]，连续的 fp32 投影权重。
        scale: [3]，fp32；分别缩放 pre、post、residual logits。
        base: [M]，fp32；按 [S, S, S * S] 分组的偏置。
        norm_weight: [H]，连续的子层 RMSNorm 权重，通常为 bf16。
        streams: 残差 stream 数 S；当前要求 S = 4。
        rms_eps: 在 S * H 维计算投影前 RMS 倒数时使用的 epsilon。
        norm_eps: 合并 stream 后，在 H 维做子层 RMSNorm 时使用的 epsilon。
        hc_eps: pre sigmoid 后的偏移量及 Sinkhorn 的稳定项。
        sinkhorn_iters: Sinkhorn 迭代次数。
        post_multiplier: post sigmoid 的乘数，默认 2.0。

    Returns:
        (layer_input, residual_mix, post_mix)：分别为 bf16 [T, H]、fp32 [T, S, S]、fp32 [T, S]。
        layer_input 已完成子层 RMSNorm；residual_mix[t, i, j] 对应输入 i 到输出 j。
        x 保持不变，留给对应 hc_post 使用。

    默认用 DeepGEMM 融合投影与平方和计算，Triton 完成 Sinkhorn、stream 合并和 RMSNorm。
    合并结果先舍入到 bf16，再做 RMSNorm。
    LIGHTLLM_DISABLE_DEEPGEMM_MHC=1 禁用 DeepGEMM，改用 PyTorch 计算投影与平方和。
    """

    assert streams == 4, "the fused mHC kernel is specialized for four streams"
    if LIGHTLLM_DISABLE_DEEPGEMM_MHC:
        # PyTorch produces one complete partition: [1, T, M] and [1, T].
        x_fp32 = x.float()
        gemm_partial = (x_fp32 @ fn.T).unsqueeze(0)
        sqrsum_partial = x_fp32.square().sum(dim=-1).unsqueeze(0)
    else:
        gemm_partial, sqrsum_partial = _hc_prenorm_projection(x, fn)

    return _finish_pre_norm(
        x,
        gemm_partial,
        sqrsum_partial,
        scale,
        base,
        norm_weight,
        streams,
        rms_eps,
        norm_eps,
        hc_eps,
        sinkhorn_iters,
        post_multiplier,
    )


def hc_post_pre_norm(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    residual_mix: torch.Tensor,
    post_mix: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    streams: int,
    rms_eps: float,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_multiplier: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete a deferred post and run the next pre + RMSNorm.

    Uses the same flat residuals and mix layout as hc_post / hc_pre_norm.
    Returns (layer_input [T, H], residual [T, S * H], residual_mix [T, S, S], post_mix [T, S]).
    Small decode batches fuse post with the projection; larger batches reuse
    hc_post and DeepGEMM. Both paths preserve bf16 rounding of the updated residual.
    """
    tokens = layer_output.shape[0]
    assert streams == 4
    if tokens <= 16 and not LIGHTLLM_DISABLE_DEEPGEMM_MHC:
        return _hc_post_pre_norm_decode(
            layer_output,
            residual,
            residual_mix,
            post_mix,
            fn,
            scale,
            base,
            norm_weight,
            streams,
            rms_eps,
            norm_eps,
            hc_eps,
            sinkhorn_iters,
            post_multiplier,
        )
    else:
        updated = hc_post(layer_output, residual, residual_mix, post_mix, streams)
        layer_input, next_residual_mix, next_post_mix = hc_pre_norm(
            updated,
            fn,
            scale,
            base,
            norm_weight,
            streams,
            rms_eps,
            norm_eps,
            hc_eps,
            sinkhorn_iters,
            post_multiplier,
        )
    return layer_input, updated, next_residual_mix, next_post_mix
