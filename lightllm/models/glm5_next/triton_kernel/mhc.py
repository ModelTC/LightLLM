# SPDX-License-Identifier: Apache-2.0

"""mHC operators used by GLM-5-Next.

The public entry points use fused Triton kernels for the small, launch-bound
mixing operations.  The explicit PyTorch implementations remain available as
correctness oracles: all mixing math is accumulated in fp32 and only the
collapsed layer input / expanded residual streams are cast back to the
activation dtype.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _hc_prepare_kernel(
    mixes,
    scale,
    base,
    pre,
    post,
    residual_mix,
    mix_stride_m: tl.constexpr,
    pre_stride_m: tl.constexpr,
    residual_stride_m: tl.constexpr,
    STREAMS: tl.constexpr,
    HC_EPS: tl.constexpr,
    POST_MULTIPLIER: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    token = tl.program_id(0)
    stream_offsets = tl.arange(0, STREAMS)
    matrix_offsets = tl.arange(0, STREAMS * STREAMS)

    pre_raw = tl.load(mixes + token * mix_stride_m + stream_offsets)
    post_raw = tl.load(mixes + token * mix_stride_m + STREAMS + stream_offsets)
    pre_values = tl.sigmoid(pre_raw * tl.load(scale) + tl.load(base + stream_offsets)) + HC_EPS
    post_values = POST_MULTIPLIER * tl.sigmoid(post_raw * tl.load(scale + 1) + tl.load(base + STREAMS + stream_offsets))

    logits = tl.load(mixes + token * mix_stride_m + 2 * STREAMS + matrix_offsets)
    logits = logits * tl.load(scale + 2) + tl.load(base + 2 * STREAMS + matrix_offsets)
    logits = tl.reshape(logits, (STREAMS, STREAMS))
    logits = logits - tl.max(logits, axis=1)[:, None]
    matrix = tl.exp(logits)
    matrix = matrix / tl.sum(matrix, axis=1)[:, None]
    matrix += HC_EPS

    # The checkpoint definition starts with a column normalization, then
    # alternates row and column normalizations for the remaining iterations.
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)
    for _ in tl.static_range(1, SINKHORN_ITERS):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + HC_EPS)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)

    tl.store(pre + token * pre_stride_m + stream_offsets, pre_values)
    tl.store(post + token * pre_stride_m + stream_offsets, post_values)
    tl.store(
        residual_mix + token * residual_stride_m + matrix_offsets,
        tl.reshape(matrix, (STREAMS * STREAMS,)),
    )


@triton.jit
def _hc_prepare_prenorm_kernel(
    gemm_partial,
    sqrsum_partial,
    scale,
    base,
    pre,
    post,
    residual_mix,
    gemm_stride_s: tl.constexpr,
    gemm_stride_m: tl.constexpr,
    sqrsum_stride_s: tl.constexpr,
    sqrsum_stride_m: tl.constexpr,
    pre_stride_m: tl.constexpr,
    residual_stride_m: tl.constexpr,
    FLATTENED_HIDDEN: tl.constexpr,
    RMS_EPS: tl.constexpr,
    STREAMS: tl.constexpr,
    HC_EPS: tl.constexpr,
    POST_MULTIPLIER: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    N_SPLITS: tl.constexpr,
):
    token = tl.program_id(0)
    stream_offsets = tl.arange(0, STREAMS)
    matrix_offsets = tl.arange(0, STREAMS * STREAMS)
    pre_raw = tl.zeros((STREAMS,), dtype=tl.float32)
    post_raw = tl.zeros((STREAMS,), dtype=tl.float32)
    matrix_raw = tl.zeros((STREAMS * STREAMS,), dtype=tl.float32)
    sqrsum = 0.0
    for split in tl.static_range(N_SPLITS):
        partial_base = gemm_partial + split * gemm_stride_s + token * gemm_stride_m
        pre_raw += tl.load(partial_base + stream_offsets)
        post_raw += tl.load(partial_base + STREAMS + stream_offsets)
        matrix_raw += tl.load(partial_base + 2 * STREAMS + matrix_offsets)
        sqrsum += tl.load(sqrsum_partial + split * sqrsum_stride_s + token * sqrsum_stride_m)
    inv_rms = tl.rsqrt(sqrsum / FLATTENED_HIDDEN + RMS_EPS)
    pre_raw *= inv_rms
    post_raw *= inv_rms
    matrix_raw *= inv_rms

    pre_values = tl.sigmoid(pre_raw * tl.load(scale) + tl.load(base + stream_offsets)) + HC_EPS
    post_values = POST_MULTIPLIER * tl.sigmoid(post_raw * tl.load(scale + 1) + tl.load(base + STREAMS + stream_offsets))

    logits = matrix_raw * tl.load(scale + 2) + tl.load(base + 2 * STREAMS + matrix_offsets)
    logits = tl.reshape(logits, (STREAMS, STREAMS))
    logits = logits - tl.max(logits, axis=1)[:, None]
    matrix = tl.exp(logits)
    matrix = matrix / tl.sum(matrix, axis=1)[:, None]
    matrix += HC_EPS
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)
    for _ in tl.static_range(1, SINKHORN_ITERS):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + HC_EPS)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)

    tl.store(pre + token * pre_stride_m + stream_offsets, pre_values)
    tl.store(post + token * pre_stride_m + stream_offsets, post_values)
    tl.store(
        residual_mix + token * residual_stride_m + matrix_offsets,
        tl.reshape(matrix, (STREAMS * STREAMS,)),
    )


@triton.jit
def _hc_pre_combine_kernel(
    x,
    pre,
    output,
    hidden: tl.constexpr,
    x_stride_m: tl.constexpr,
    pre_stride_m: tl.constexpr,
    out_stride_m: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for stream in tl.static_range(STREAMS):
        residual = tl.load(
            x + token * x_stride_m + stream * hidden + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        pre_value = tl.load(pre + token * pre_stride_m + stream)
        accumulator += residual * pre_value
    tl.store(
        output + token * out_stride_m + hidden_offsets,
        accumulator,
        mask=hidden_mask,
    )


@triton.jit
def _hc_pre_combine_norm_kernel(
    x,
    pre,
    norm_weight,
    output,
    hidden: tl.constexpr,
    x_stride_m: tl.constexpr,
    pre_stride_m: tl.constexpr,
    out_stride_m: tl.constexpr,
    STREAMS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for stream in tl.static_range(STREAMS):
        residual = tl.load(
            x + token * x_stride_m + stream * hidden + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        pre_value = tl.load(pre + token * pre_stride_m + stream)
        accumulator += residual * pre_value

    # hc_pre returns bf16 before the following RMSNorm in the checkpoint
    # definition.  Preserve that rounding point while keeping both operations
    # in one kernel.
    rounded = accumulator.to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(rounded * rounded, axis=0) / hidden
    inv_rms = tl.rsqrt(variance + NORM_EPS)
    weight = tl.load(norm_weight + hidden_offsets, mask=hidden_mask, other=0.0)
    tl.store(
        output + token * out_stride_m + hidden_offsets,
        rounded * inv_rms * weight,
        mask=hidden_mask,
    )


@triton.jit
def _hc_post_4stream_kernel(
    layer_output,
    residual,
    residual_mix,
    post_mix,
    output,
    hidden: tl.constexpr,
    layer_stride_m: tl.constexpr,
    residual_stride_m: tl.constexpr,
    mix_stride_m: tl.constexpr,
    post_stride_m: tl.constexpr,
    out_stride_m: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden

    # GLM-5 always uses four mHC streams.  Compute all four outputs in one
    # program so the layer output and residual streams are read only once.
    # The previous output-stream grid reread each residual stream four times;
    # that becomes bandwidth-bound for large prefills.
    layer_value = tl.load(
        layer_output + token * layer_stride_m + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    residual_base = residual + token * residual_stride_m + hidden_offsets
    residual_0 = tl.load(residual_base, mask=hidden_mask, other=0.0).to(tl.float32)
    residual_1 = tl.load(residual_base + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
    residual_2 = tl.load(residual_base + 2 * hidden, mask=hidden_mask, other=0.0).to(tl.float32)
    residual_3 = tl.load(residual_base + 3 * hidden, mask=hidden_mask, other=0.0).to(tl.float32)

    post_base = post_mix + token * post_stride_m
    mix_base = residual_mix + token * mix_stride_m
    accumulator_0 = layer_value * tl.load(post_base)
    accumulator_1 = layer_value * tl.load(post_base + 1)
    accumulator_2 = layer_value * tl.load(post_base + 2)
    accumulator_3 = layer_value * tl.load(post_base + 3)

    accumulator_0 += residual_0 * tl.load(mix_base)
    accumulator_0 += residual_1 * tl.load(mix_base + 4)
    accumulator_0 += residual_2 * tl.load(mix_base + 8)
    accumulator_0 += residual_3 * tl.load(mix_base + 12)
    accumulator_1 += residual_0 * tl.load(mix_base + 1)
    accumulator_1 += residual_1 * tl.load(mix_base + 5)
    accumulator_1 += residual_2 * tl.load(mix_base + 9)
    accumulator_1 += residual_3 * tl.load(mix_base + 13)
    accumulator_2 += residual_0 * tl.load(mix_base + 2)
    accumulator_2 += residual_1 * tl.load(mix_base + 6)
    accumulator_2 += residual_2 * tl.load(mix_base + 10)
    accumulator_2 += residual_3 * tl.load(mix_base + 14)
    accumulator_3 += residual_0 * tl.load(mix_base + 3)
    accumulator_3 += residual_1 * tl.load(mix_base + 7)
    accumulator_3 += residual_2 * tl.load(mix_base + 11)
    accumulator_3 += residual_3 * tl.load(mix_base + 15)

    output_base = output + token * out_stride_m + hidden_offsets
    tl.store(output_base, accumulator_0, mask=hidden_mask)
    tl.store(output_base + hidden, accumulator_1, mask=hidden_mask)
    tl.store(output_base + 2 * hidden, accumulator_2, mask=hidden_mask)
    tl.store(output_base + 3 * hidden, accumulator_3, mask=hidden_mask)


def hc_expand(x: torch.Tensor, streams: int) -> torch.Tensor:
    """Expand ``[tokens, hidden]`` into flattened residual streams."""

    assert x.ndim == 2
    return x.unsqueeze(1).expand(-1, streams, -1).reshape(x.shape[0], -1)


def hc_contract(x: torch.Tensor, streams: int) -> torch.Tensor:
    """Contract flattened residual streams by taking their mean."""

    assert x.ndim == 2 and x.shape[-1] % streams == 0
    return x.view(x.shape[0], streams, -1).mean(dim=1)


def hc_pre_reference(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    streams: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_multiplier: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC pre-mixes.

    Returns ``(layer_input, residual_mix, post_mix)``.  ``x`` and
    ``layer_input`` use the activation dtype; both mix tensors are fp32.
    """

    assert x.ndim == 2 and x.shape[-1] % streams == 0
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    residual = x.view(tokens, streams, hidden)

    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_fp32, fn) * inv_rms

    pre_raw = mixes[:, :streams]
    post_raw = mixes[:, streams : 2 * streams]
    residual_raw = mixes[:, 2 * streams :].view(tokens, streams, streams)

    pre = torch.sigmoid(pre_raw * scale[0] + base[:streams]) + hc_eps
    post = post_multiplier * torch.sigmoid(post_raw * scale[1] + base[streams : 2 * streams])
    residual_mix = (residual_raw * scale[2] + base[2 * streams :].view(streams, streams)).softmax(dim=-1)
    residual_mix = residual_mix + hc_eps
    residual_mix = residual_mix / (residual_mix.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        residual_mix = residual_mix / (residual_mix.sum(dim=-1, keepdim=True) + hc_eps)
        residual_mix = residual_mix / (residual_mix.sum(dim=-2, keepdim=True) + hc_eps)

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(x.dtype)
    return layer_input, residual_mix, post


def hc_post_reference(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    residual_mix: torch.Tensor,
    post_mix: torch.Tensor,
    streams: int,
) -> torch.Tensor:
    """Mix a sublayer output back into the flattened residual streams."""

    tokens, hidden = layer_output.shape
    residual_3d = residual.view(tokens, streams, hidden)
    mixed_residual = (residual_mix.unsqueeze(-1) * residual_3d.float().unsqueeze(2)).sum(dim=1)
    out = post_mix.unsqueeze(-1) * layer_output.float().unsqueeze(1) + mixed_residual
    return out.to(layer_output.dtype).reshape(tokens, streams * hidden)


def hc_pre(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    streams: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_multiplier: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mHC pre-mixes with fused Sinkhorn and residual combining."""

    assert x.ndim == 2 and x.shape[-1] % streams == 0
    assert streams == 4, "the fused GLM-5 mHC kernel is specialized for four streams"
    assert x.is_contiguous() and fn.is_contiguous()
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams

    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_fp32, fn) * inv_rms

    pre = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
    post = torch.empty_like(pre)
    residual_mix = torch.empty((tokens, streams, streams), dtype=torch.float32, device=x.device)
    _hc_prepare_kernel[(tokens,)](
        mixes,
        scale,
        base,
        pre,
        post,
        residual_mix,
        mixes.stride(0),
        pre.stride(0),
        residual_mix.stride(0),
        STREAMS=streams,
        HC_EPS=hc_eps,
        POST_MULTIPLIER=post_multiplier,
        SINKHORN_ITERS=sinkhorn_iters,
        num_warps=1,
    )

    layer_input = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_h = min(triton.next_power_of_2(hidden), 1024)
    _hc_pre_combine_kernel[(tokens, triton.cdiv(hidden, block_h))](
        x,
        pre,
        layer_input,
        hidden=hidden,
        x_stride_m=x.stride(0),
        pre_stride_m=pre.stride(0),
        out_stride_m=layer_input.stride(0),
        STREAMS=streams,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return layer_input, residual_mix, post


def _compute_prenorm_splits(tokens: int, flattened_hidden: int, device: torch.device) -> int:
    grid_size = triton.cdiv(tokens, 64)
    k_blocks = triton.cdiv(flattened_hidden, 64)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return max(1, min(sms // max(grid_size, 1), k_blocks // 4))


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
    """Fuse mHC pre-mixing with its immediately following RMSNorm.

    DeepGEMM computes the fp32 projection and residual square sum in one
    split-K kernel.  Triton then reduces those partials, runs Sinkhorn, forms
    the bf16 residual collapse, and applies RMSNorm.
    """

    assert x.ndim == 2 and x.shape[-1] % streams == 0
    assert streams == 4, "the fused GLM-5 mHC kernel is specialized for four streams"
    assert x.dtype == torch.bfloat16 and fn.dtype == torch.float32
    assert x.is_contiguous() and fn.is_contiguous() and norm_weight.is_contiguous()
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams

    try:
        import deep_gemm

        prenorm_gemm = deep_gemm.tf32_hc_prenorm_gemm
    except (AttributeError, ImportError):
        from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import (
            rmsnorm_forward,
        )

        layer_input, residual_mix, post = hc_pre(
            x,
            fn,
            scale,
            base,
            streams,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            post_multiplier,
        )
        layer_input = rmsnorm_forward(layer_input, weight=norm_weight, eps=norm_eps)
        return layer_input, residual_mix, post

    mix_size = (2 + streams) * streams
    n_splits = _compute_prenorm_splits(tokens, flattened_hidden, x.device)
    gemm_partial = torch.empty((n_splits, tokens, mix_size), dtype=torch.float32, device=x.device)
    sqrsum_partial = torch.empty((n_splits, tokens), dtype=torch.float32, device=x.device)
    prenorm_gemm(x, fn, gemm_partial, sqrsum_partial, n_splits)

    pre = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
    post = torch.empty_like(pre)
    residual_mix = torch.empty((tokens, streams, streams), dtype=torch.float32, device=x.device)
    _hc_prepare_prenorm_kernel[(tokens,)](
        gemm_partial,
        sqrsum_partial,
        scale,
        base,
        pre,
        post,
        residual_mix,
        gemm_partial.stride(0),
        gemm_partial.stride(1),
        sqrsum_partial.stride(0),
        sqrsum_partial.stride(1),
        pre.stride(0),
        residual_mix.stride(0),
        FLATTENED_HIDDEN=flattened_hidden,
        RMS_EPS=rms_eps,
        STREAMS=streams,
        HC_EPS=hc_eps,
        POST_MULTIPLIER=post_multiplier,
        SINKHORN_ITERS=sinkhorn_iters,
        N_SPLITS=n_splits,
        num_warps=1,
    )

    layer_input = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(hidden)
    _hc_pre_combine_norm_kernel[(tokens,)](
        x,
        pre,
        norm_weight,
        layer_input,
        hidden=hidden,
        x_stride_m=x.stride(0),
        pre_stride_m=pre.stride(0),
        out_stride_m=layer_input.stride(0),
        STREAMS=streams,
        NORM_EPS=norm_eps,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return layer_input, residual_mix, post


def hc_post(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    residual_mix: torch.Tensor,
    post_mix: torch.Tensor,
    streams: int,
) -> torch.Tensor:
    """Mix a sublayer output into residual streams with one Triton launch."""

    tokens, hidden = layer_output.shape
    assert streams == 4, "the fused GLM-5 mHC kernel is specialized for four streams"
    assert layer_output.is_contiguous() and residual.is_contiguous()
    output = torch.empty(
        (tokens, streams * hidden),
        dtype=layer_output.dtype,
        device=layer_output.device,
    )
    block_h = min(triton.next_power_of_2(hidden), 1024)
    _hc_post_4stream_kernel[(tokens, triton.cdiv(hidden, block_h))](
        layer_output,
        residual,
        residual_mix,
        post_mix,
        output,
        hidden=hidden,
        layer_stride_m=layer_output.stride(0),
        residual_stride_m=residual.stride(0),
        mix_stride_m=residual_mix.stride(0),
        post_stride_m=post_mix.stride(0),
        out_stride_m=output.stride(0),
        BLOCK_H=block_h,
        num_warps=8,
    )
    return output
