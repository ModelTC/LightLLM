# SPDX-License-Identifier: Apache-2.0
"""Compact speculative state for GDN and KDA.

Verify retains raw inputs rather than full SSM snapshots. Commit re-executes the
accepted prefix with the same per-token rounding as the recurrent baseline.
"""

import torch
import triton
import triton.language as tl

from .ssm_autotune import configure_cache


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
    key_slot = (layer * SLOTS + req) * KH + hv // (HV // KH)
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


class CompactSSMCache:
    def __init__(
        self, state, verify_width, activation_dtype, *, num_key_heads=None, kda=False, lower_bound=-5.0, run_config=None
    ):
        self.state = state
        self.kda = kda
        self.lower_bound = lower_bound
        self.verify_width = verify_width
        # Verify and accepted-state reconstruction must share the same layout.
        self._config_is_fixed = run_config is not None
        self.batch_configs = {}
        self.run_config = dict(run_config or {"BV": 8, "num_warps": 1})
        assert self.run_config["BV"] in (8, 16, 32, 64, 128)
        assert self.run_config["num_warps"] in (1, 2, 4, 8)
        layers, slots, hv, k, v = state.shape
        self.num_key_heads = hv if num_key_heads is None else num_key_heads
        assert self.num_key_heads > 0 and hv % self.num_key_heads == 0
        self.hold = slots - 1
        shape = (layers, slots, hv, verify_width)
        self.keys = torch.empty(
            (layers, slots, self.num_key_heads, verify_width, k), device=state.device, dtype=activation_dtype
        )
        self.values = torch.empty((*shape, v), device=state.device, dtype=activation_dtype)
        self.decays = torch.empty((*shape, k) if kda else shape, device=state.device, dtype=torch.float32)
        self.betas = torch.empty(shape, device=state.device, dtype=torch.float32)

    def clear_history(self, req):
        # Verify overwrites scratch before accepting updates; no history spans rounds.
        pass

    def prepare_decode(self, reqs, cu_seqlens=None):
        # Compact mode has no history spanning rounds.
        return None

    def merge_accepted_updates(self, reqs):
        # accept_updates already merged the accepted prefix into SSM state.
        pass

    def snapshot_accepted_state(self, req_idx):
        return self.state[:, req_idx]

    def get_run_config(self, q, k, v, a, b, a_log, bias, cu_seqlens=None):
        return configure_cache(self, "gdn", q, k, v, a, b, a_log, bias, cu_seqlens)

    def forward(self, layer, q, k, v, a, b, a_log, bias, reqs, cu_seqlens=None, run_config=None):
        assert cu_seqlens is not None, "compact replay is only used for speculative verify"
        config = run_config or self.get_run_config(q, k, v, a, b, a_log, bias, cu_seqlens)
        _, slots, hv, kd, vd = self.state.shape
        assert self.num_key_heads in (q.shape[-2], hv)
        out = torch.empty_like(v)
        # Match native reduction layouts: changing BV can alter BF16 rounding.
        bv = config["BV"]
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
            self.num_key_heads,
            kd,
            vd,
            self.verify_width,
            self.hold,
            triton.next_power_of_2(kd),
            bv,
            False,
            self.kda,
            self.lower_bound,
            num_warps=config["num_warps"],
            num_stages=config.get("num_stages", 3),
        )
        return out

    def accept_updates(self, reqs, accepted, run_config=None):
        """Merge this round's prefix; accepted contains per-request last accepted indexes."""
        assert run_config is not None or not self.batch_configs, "compact acceptance requires the verify configuration"
        layers, slots, hv, kd, vd = self.state.shape
        config = run_config or self.run_config
        bv = config["BV"]
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
            self.num_key_heads,
            kd,
            vd,
            self.verify_width,
            self.hold,
            triton.next_power_of_2(kd),
            bv,
            True,
            self.kda,
            self.lower_bound,
            num_warps=config["num_warps"],
            num_stages=config.get("num_stages", 3),
        )
