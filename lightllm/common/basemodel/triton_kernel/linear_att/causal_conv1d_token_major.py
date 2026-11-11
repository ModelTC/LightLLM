# SPDX-License-Identifier: Apache-2.0

"""Width-four BF16 Prefill convolution over token-major QKV."""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv(
    X,
    W,
    STATE,
    START,
    SLOT,
    WARM,
    Y,
    C: tl.constexpr,
    XS: tl.constexpr,
    SS: tl.constexpr,
    SC: tl.constexpr,
    SL: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    req = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    first = tl.load(START + req)
    length = tl.load(START + req + 1) - first
    slot = tl.load(SLOT + req)
    warm = tl.load(WARM + req)
    valid = (t[:, None] < length) & (c[None, :] < C)
    if tl.program_id(1) * BT < length:
        if slot == -1:
            data = tl.load(X + (first + t[:, None]) * XS + c[None, :], valid, other=0)
            tl.store(Y + (first + t[:, None]) * C + c[None, :], data, valid)
        else:
            acc = tl.full((BT, BC), 0.0, tl.float32)
            for j in tl.static_range(4):
                pos = t + j - 3
                value = tl.load(X + (first + pos[:, None]) * XS + c[None, :], valid & (pos[:, None] >= 0), other=0).to(
                    tl.float32
                )
                if j < 3:
                    if warm and tl.program_id(1) == 0:
                        past = tl.load(
                            STATE + slot * SS + c[None, :] * SC + (pos[:, None] + 3) * SL,
                            valid & (pos[:, None] < 0),
                            other=0,
                        ).to(tl.float32)
                        value = tl.where(pos[:, None] < 0, past, value)
                w = tl.load(W + c * 4 + j, c < C, other=0).to(tl.float32)
                acc = tl.fma(value, w[None, :], acc)
            out = acc / (1.0 + tl.exp(-acc))
            tl.store(Y + (first + t[:, None]) * C + c[None, :], out, valid)


@triton.jit
def _store_state(
    X,
    STATE,
    START,
    SLOT,
    WARM,
    C: tl.constexpr,
    XS: tl.constexpr,
    SS: tl.constexpr,
    SC: tl.constexpr,
    SL: tl.constexpr,
    BC: tl.constexpr,
):
    req = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    first = tl.load(START + req)
    length = tl.load(START + req + 1) - first
    slot = tl.load(SLOT + req)
    warm = tl.load(WARM + req)
    if slot >= 0 and length > 0:
        h = tl.arange(0, 4)
        t = length - 3 + h
        valid = (c[None, :] < C) & (h[:, None] < 3)
        new = tl.load(X + (first + t[:, None]) * XS + c[None, :], valid & (t[:, None] >= 0), other=0)
        old = tl.load(
            STATE + slot * SS + c[None, :] * SC + (t[:, None] + 3) * SL, valid & (t[:, None] < 0) & warm, other=0
        )
        tl.store(STATE + slot * SS + c[None, :] * SC + h[:, None] * SL, tl.where(t[:, None] >= 0, new, old), valid)


def causal_conv1d_prefill_token_major(x, weight, states, starts, slots, warm, max_len):
    """Return token-major output; use old states until all convolution tiles finish."""
    c = x.shape[1]
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    bt, bc = 16, 256
    _conv[(slots.numel(), triton.cdiv(max_len, bt), triton.cdiv(c, bc))](
        x, weight, states, starts, slots, warm, y, c, x.stride(0), *states.stride(), bt, bc, num_warps=4
    )
    _store_state[(slots.numel(), triton.cdiv(c, bc))](
        x, states, starts, slots, warm, c, x.stride(0), *states.stride(), bc, num_warps=4
    )
    return y
