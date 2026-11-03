# SPDX-License-Identifier: Apache-2.0
"""Compact speculative state for GDN and KDA.

Verify retains raw inputs rather than full SSM snapshots. Commit re-executes the
accepted prefix with the same per-token rounding as the recurrent baseline.
"""

import triton
import triton.language as tl


@triton.jit
def _compact(
    Q,
    Kp,
    Vp,
    A,
    B,
    Alog,
    Bias,
    State,
    Keys,
    Values,
    Decays,
    Betas,
    Reqs,
    Cu,
    Accepted,
    Out,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    SLOTS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    KH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    WIDTH: tl.constexpr,
    HOLD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    COMMIT: tl.constexpr,
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
):
    iv, row, lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(Reqs + row).to(tl.int64)
    linear_layer_idx, hv = lh // HV, lh % HV
    if COMMIT:
        start = 0
        end = tl.load(Accepted + req) + 1 if req != HOLD else 0
    else:
        start, end = tl.load(Cu + row), tl.load(Cu + row + 1)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    if req == HOLD or end == start:
        if not COMMIT:
            for t in range(start, end):
                tl.store(Out + (t * HV + hv) * V + vv, 0, vv < V)
        return
    slot = (linear_layer_idx * SLOTS + req) * HV + hv
    key_slot = (linear_layer_idx * SLOTS + req) * KH + hv // (HV // KH)
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    for t in range(start, end):
        record = slot * WIDTH + t - start
        key_record = key_slot * WIDTH + t - start
        if COMMIT:
            k = tl.load(Keys + key_record * K + kk, kk < K, 0).to(tl.float32)
            v = tl.load(Values + record * V + vv, vv < V, 0).to(tl.float32)
            if KDA:
                decay = tl.load(Decays + record * K + kk, kk < K, 1)[:, None]
            else:
                decay = tl.load(Decays + record)
            beta = tl.load(Betas + record)
        else:
            h = hv // (HV // H)
            k = tl.load(Kp + t * SK + h * K + kk, kk < K, 0).to(tl.float32)
            v = tl.load(Vp + t * SV + hv * V + vv, vv < V, 0).to(tl.float32)
            log_a = tl.load(Alog + hv).to(tl.float32)
            if KDA:
                x = tl.load(A + t * SA + hv * K + kk, kk < K, 0).to(tl.float32)
                x += tl.load(Bias + hv * K + kk, kk < K, 0).to(tl.float32)
                decay = tl.exp(LOWER * tl.sigmoid(tl.exp(log_a) * x))[:, None]
            else:
                x = tl.load(A + t * SA + hv).to(tl.float32) + tl.load(Bias + hv).to(tl.float32)
                g = -tl.exp(log_a) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
                decay = tl.exp(g)
            beta = tl.sigmoid(tl.load(B + t * SB + hv).to(tl.float32))
            tl.store(Values + record * V + vv, v, vv < V)
            if iv == 0:
                if hv % (HV // KH) == 0:
                    tl.store(Keys + key_record * K + kk, k, kk < K)
                tl.store(Betas + record, beta)
                if KDA:
                    tl.store(Decays + record * K + kk, tl.sum(decay, 1), kk < K)
                else:
                    tl.store(Decays + record, decay)
        k = k / tl.sqrt(tl.sum(k * k) + 1.0e-6)
        state *= decay
        d = (v - tl.sum(state * k[:, None], 0)) * beta
        state += k[:, None] * d[None, :]
        if not COMMIT:
            q = tl.load(Q + t * SQ + h * K + kk, kk < K, 0).to(tl.float32)
            q = q / tl.sqrt(tl.sum(q * q) + 1.0e-6) * (K ** -0.5)
            tl.store(Out + (t * HV + hv) * V + vv, tl.sum(state * q[:, None], 0), vv < V)
        state = state.to(State.dtype.element_ty).to(tl.float32)
    if COMMIT:
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
