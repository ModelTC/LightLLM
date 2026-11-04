# SPDX-License-Identifier: Apache-2.0

"""Learned, sigmoid-gated contraction of residual streams."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from lightllm.common.triton_utils.autotuner import autotune
from .configs import head_configs, head_key


@triton.jit
def _hc_head_kernel(
    residual,
    fn,
    scale,
    base,
    output,
    HIDDEN: tl.constexpr,
    STREAMS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    streams = tl.arange(0, STREAMS)
    squared = tl.zeros((BLOCK_K,), dtype=tl.float32)
    projections = tl.zeros((STREAMS, BLOCK_K), dtype=tl.float32)
    for start in range(triton.cdiv(STREAMS * HIDDEN, BLOCK_K)):
        columns = start * BLOCK_K + offsets
        values = tl.load(residual + token * STREAMS * HIDDEN + columns, columns < STREAMS * HIDDEN, 0.0)
        values = values.to(tl.float32)
        squared += values * values
        weights = tl.load(
            fn + streams[:, None] * STREAMS * HIDDEN + columns[None, :],
            columns[None, :] < STREAMS * HIDDEN,
            0.0,
        )
        projections += weights * values[None, :]
    inv_rms = tl.rsqrt(tl.sum(squared, 0) / (STREAMS * HIDDEN) + RMS_EPS)
    gates = tl.sigmoid(tl.sum(projections, 1) * inv_rms * tl.load(scale) + tl.load(base + streams)) + HC_EPS
    for start in range(triton.cdiv(HIDDEN, BLOCK_K)):
        columns = start * BLOCK_K + offsets
        values = tl.load(
            residual + token * STREAMS * HIDDEN + streams[:, None] * HIDDEN + columns[None, :],
            columns[None, :] < HIDDEN,
            0.0,
        ).to(tl.float32)
        combined = tl.sum(values * gates[:, None], axis=0)
        tl.store(output + token * HIDDEN + columns, combined, columns < HIDDEN)


@autotune(
    kernel_name="mhc_head:v1",
    configs_gen_func=head_configs,
    static_key_func=head_key,
    run_key_func=lambda x: x.shape[0],
    benchmark_repeats=32,
    benchmark_retries=5,
)
def hc_head(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    streams: int,
    rms_eps: float,
    hc_eps: float,
    out: torch.Tensor | None = None,
    run_config: dict = None,
) -> torch.Tensor:
    """Collapse [T, S * H] to [T, H] using learned gates, without a final RMSNorm.

    fn: fp32 [S, S * H], scale: fp32 [1], base: fp32 [S]. Unlike
    hc_contract's mean reduction, gates depend on the RMS-normalized residual.
    """
    assert streams == 4, "the fused mHC head is specialized for four streams"
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    if out is None:
        out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if run_config is None:
        run_config = {"BLOCK_K": 4096 if tokens <= 128 else 1024, "num_warps": 8 if tokens <= 128 else 4}
    _hc_head_kernel[(tokens,)](
        x,
        fn,
        scale,
        base,
        out,
        HIDDEN=hidden,
        STREAMS=streams,
        RMS_EPS=rms_eps,
        HC_EPS=hc_eps,
        BLOCK_K=run_config["BLOCK_K"],
        num_warps=run_config["num_warps"],
    )
    return out
