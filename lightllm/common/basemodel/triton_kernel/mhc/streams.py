# SPDX-License-Identifier: Apache-2.0

"""Expansion and contraction of flattened residual streams."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def hc_expand(x: torch.Tensor, streams: int) -> torch.Tensor:
    """Copy each token to S residual streams.

    x: [T, H]. Return [T, S * H] with the same dtype and device.
    Each stream has the same embedding.
    """

    assert x.ndim == 2
    return x.unsqueeze(1).expand(-1, streams, -1).reshape(x.shape[0], streams * x.shape[1])


def hc_contract(x: torch.Tensor, streams: int) -> torch.Tensor:
    """Take the mean of S residual streams.

    x: [T, S * H]. Return [T, H] with the same dtype and device.
    Models with learned gates use hc_head instead.
    """

    assert x.ndim == 2 and x.shape[-1] % streams == 0
    return x.view(x.shape[0], streams, x.shape[1] // streams).mean(dim=1)


@triton.jit
def _hc_combine_kernel(
    residual,
    gates,
    output,
    cu_q_lens,
    tokens,
    residual_row_stride,
    residual_col_stride,
    gate_row_stride,
    gate_col_stride,
    HIDDEN: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
    PREFILL_LAST_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    input_row = row.to(tl.int64)
    if PREFILL_LAST_ROWS:
        last = tl.load(cu_q_lens + row + 1) - 1
        input_row = tl.where(last < 0, tokens - 1, last).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    residual_row = residual + input_row * residual_row_stride
    gate_row = gates + input_row * gate_row_stride
    combined = tl.zeros((BLOCK,), dtype=tl.float32)
    for stream in tl.static_range(STREAMS):
        values = tl.load(residual_row + (stream * HIDDEN + cols) * residual_col_stride, cols < HIDDEN, 0).to(tl.float32)
        gate = tl.load(gate_row + stream * gate_col_stride)
        # Round the product before adding, as in the unfused PyTorch reduction.
        combined = libdevice.add_rn(combined, libdevice.mul_rn(values, gate))
    tl.store(output + row.to(tl.int64) * HIDDEN + cols, combined, cols < HIDDEN)


def hc_combine(x: torch.Tensor, gates: torch.Tensor, *, cu_q_lens: torch.Tensor | None = None) -> torch.Tensor:
    """Combine flattened streams, selecting just the final row of each query when given offsets."""
    tokens, flattened_hidden = x.shape
    streams = gates.shape[1]
    hidden = flattened_hidden // streams
    output_rows = tokens if cu_q_lens is None else cu_q_lens.shape[0] - 1
    out = torch.empty((output_rows, hidden), dtype=x.dtype, device=x.device)
    block = 512
    _hc_combine_kernel[(output_rows, triton.cdiv(hidden, block))](
        x,
        gates,
        out,
        cu_q_lens,
        tokens,
        *x.stride(),
        *gates.stride(),
        hidden,
        streams,
        block,
        PREFILL_LAST_ROWS=cu_q_lens is not None,
        enable_fp_fusion=False,
    )
    return out
