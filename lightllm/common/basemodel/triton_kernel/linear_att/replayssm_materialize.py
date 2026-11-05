# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM accepted-state materialization kernel."""

import triton
import triton.language as tl


@triton.jit
def _materialize(
    State,
    Output,
    RawKeys,
    RawValues,
    Gates,
    Betas,
    Cursors,
    Reqs,
    SLOTS: tl.constexpr,
    HV: tl.constexpr,
    KH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    HOLD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SNAPSHOT: tl.constexpr,
    KDA: tl.constexpr,
    OUTPUT_STRIDES: tl.constexpr = None,
):
    iv, row, lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(Reqs + row).to(tl.int64)
    if req == HOLD:
        return
    cursor = tl.load(Cursors + req)
    n = cursor % (2 * L)
    base = (cursor // (2 * L)) * L
    if n == 0 and not SNAPSHOT:
        return
    linear_layer_idx, head = lh // HV, lh % HV
    slot = (linear_layer_idx * SLOTS + req) * HV + head
    key_slot = (linear_layer_idx * SLOTS + req) * KH + head // (HV // KH)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    for j in range(n):
        raw_k = tl.load(RawKeys + (key_slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
        raw_v = tl.load(RawValues + (slot * (2 * L) + base + j) * V + vv, vv < V, 0).to(tl.float32)
        if KDA:
            g = tl.load(Gates + (slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
        else:
            g = tl.load(Gates + slot * (2 * L) + base + j).to(tl.float32)
        beta = tl.load(Betas + slot * (2 * L) + base + j).to(tl.float32)
        raw_k /= tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
        state *= tl.exp(g)[:, None] if KDA else tl.exp(g)
        d = beta * (raw_v - tl.sum(state * raw_k[:, None], 0))
        state += raw_k[:, None] * d[None, :]
    if OUTPUT_STRIDES is not None:
        target = tl.load(Output + row).to(tl.pointer_type(State.dtype.element_ty))
        out = (
            target
            + linear_layer_idx.to(tl.int64) * OUTPUT_STRIDES[0]
            + head * OUTPUT_STRIDES[1]
            + kk[:, None] * OUTPUT_STRIDES[2]
            + vv[None, :] * OUTPUT_STRIDES[3]
        )
    else:
        out_slot = linear_layer_idx * HV + head if SNAPSHOT else slot
        out = Output + out_slot * K * V + kk[:, None] * V + vv[None, :]
    tl.store(out, state, (kk[:, None] < K) & (vv[None, :] < V))
