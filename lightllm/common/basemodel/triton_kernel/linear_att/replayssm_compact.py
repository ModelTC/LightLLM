# SPDX-License-Identifier: Apache-2.0
"""Compact speculative state for GDN/BF16 and KDA.

Verify retains raw inputs rather than full SSM snapshots. Commit re-executes the
accepted prefix with the same per-token rounding as the recurrent baseline.
KDA follows GLM-5.3-Flash's bounded per-key gate and rsqrt normalization.
"""

import torch
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
    layer, hv = lh // HV, lh % HV
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
    slot = (layer * SLOTS + req) * HV + hv
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    for t in range(start, end):
        record = slot * WIDTH + t - start
        if COMMIT:
            k = tl.load(Keys + record * K + kk, kk < K, 0).to(tl.float32)
            v = tl.load(Values + record * V + vv, vv < V, 0).to(tl.float32)
            decay = tl.load(Decays + record * K + kk, kk < K, 0) if KDA else tl.load(Decays + record)
            beta = tl.load(Betas + record)
        else:
            h = hv // (HV // H)
            k = tl.load(Kp + t * SK + h * K + kk, kk < K, 0).to(tl.float32)
            v = tl.load(Vp + t * SV + hv * V + vv, vv < V, 0).to(tl.float32)
            log_a = tl.load(Alog + hv).to(tl.float32)
            if KDA:
                bias = tl.load(Bias + hv * K + kk, kk < K, 0).to(tl.float32)
                gate = tl.load(A + t * SA + hv * K + kk, kk < K, 0).to(tl.float32)
                decay = tl.exp(LOWER * tl.sigmoid(tl.exp(log_a) * (gate + bias)))
            else:
                x = tl.load(A + t * SA + hv).to(tl.float32) + tl.load(Bias + hv).to(tl.float32)
                g = -tl.exp(log_a) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
                decay = tl.exp(g)
            beta = tl.sigmoid(tl.load(B + t * SB + hv).to(tl.float32))
            tl.store(Values + record * V + vv, v, vv < V)
            if iv == 0:
                tl.store(Keys + record * K + kk, k, kk < K)
                tl.store(Betas + record, beta)
                if KDA:
                    tl.store(Decays + record * K + kk, decay, kk < K)
                else:
                    tl.store(Decays + record, decay)
        if KDA:
            k = k * tl.rsqrt(tl.sum(k * k) + 1.0e-6)
            state *= decay[:, None]
        else:
            k = k / tl.sqrt(tl.sum(k * k) + 1.0e-6)
            state *= decay
        d = (v - tl.sum(state * k[:, None], 0)) * beta
        state += k[:, None] * d[None, :]
        if not COMMIT:
            q = tl.load(Q + t * SQ + h * K + kk, kk < K, 0).to(tl.float32)
            if KDA:
                q = q * (tl.rsqrt(tl.sum(q * q) + 1.0e-6) * (K**-0.5))
            else:
                q = q / tl.sqrt(tl.sum(q * q) + 1.0e-6) * (K**-0.5)
            tl.store(Out + (t * HV + hv) * V + vv, tl.sum(state * q[:, None], 0), vv < V)
        state = state.to(State.dtype.element_ty).to(tl.float32)
    if COMMIT:
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))


class CompactSSMCache:
    def __init__(self, state, verify_width, activation_dtype, kind="gdn", lower_bound=-5.0):
        assert kind in ("gdn", "kda")
        self.state = state
        self.verify_width = verify_width
        self.kind = kind
        self.lower_bound = lower_bound
        layers, slots, hv, k, v = state.shape
        self.hold = slots - 1
        shape = (layers, slots, hv, verify_width)
        self.keys = torch.empty((*shape, k), device=state.device, dtype=activation_dtype)
        self.values = torch.empty((*shape, v), device=state.device, dtype=activation_dtype)
        self.decays = torch.empty((*shape, k) if kind == "kda" else shape, device=state.device, dtype=torch.float32)
        self.betas = torch.empty(shape, device=state.device, dtype=torch.float32)

    def reset(self, req):
        # Scratch is overwritten by verify before commit; it has no accepted history.
        pass

    def positions(self, reqs):
        return None

    def materialize(self, reqs):
        # Commit always leaves a canonical checkpoint.
        pass

    def forward(self, layer, q, k, v, a, b, a_log, bias, reqs, positions, cu_seqlens=None):
        assert cu_seqlens is not None, "compact replay is only used for speculative verify"
        _, slots, hv, kd, vd = self.state.shape
        out = torch.empty_like(v)
        # Match native reduction layouts: changing BV can alter BF16 rounding.
        bv = 32 if self.kind == "kda" else 8
        _compact[(triton.cdiv(vd, bv), reqs.numel(), hv)](
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            self.state[layer],
            self.keys[layer],
            self.values[layer],
            self.decays[layer],
            self.betas[layer],
            reqs,
            cu_seqlens,
            None,
            out,
            q.stride(1),
            k.stride(1),
            v.stride(1),
            a.stride(0),
            b.stride(0),
            slots,
            q.shape[-2],
            hv,
            kd,
            vd,
            self.verify_width,
            self.hold,
            triton.next_power_of_2(kd),
            bv,
            False,
            self.kind == "kda",
            self.lower_bound,
            num_warps=4 if self.kind == "kda" else 1,
        )
        return out

    def commit(self, reqs, accepted):
        layers, slots, hv, kd, vd = self.state.shape
        bv = 32 if self.kind == "kda" else 8
        _compact[(triton.cdiv(vd, bv), reqs.numel(), layers * hv)](
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.state,
            self.keys,
            self.values,
            self.decays,
            self.betas,
            reqs,
            None,
            accepted,
            None,
            0,
            0,
            0,
            0,
            0,
            slots,
            hv,
            hv,
            kd,
            vd,
            self.verify_width,
            self.hold,
            triton.next_power_of_2(kd),
            bv,
            True,
            self.kind == "kda",
            self.lower_bound,
            num_warps=4 if self.kind == "kda" else 1,
        )
