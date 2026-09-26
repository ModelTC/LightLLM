# SPDX-License-Identifier: Apache-2.0
"""GDN ReplaySSM in LightLLM's [K, V] state layout.

The checkpoint contains only folded tokens. Derived records reconstruct verify
outputs, while raw inputs replay the accepted suffix into the checkpoint. Verify
may overwrite the uncommitted suffix; only acceptance advances the cursor.
Algorithm reference: https://dao-lab.ai/blog/2026/replayssm/
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _positions(
    cursors,
    reqs,
    out,
    accepted,
    N: tl.constexpr,
    HOLD: tl.constexpr,
    L: tl.constexpr,
    WIDTH: tl.constexpr,
    COMMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(reqs + i, i < N, HOLD)
    valid = (i < N) & (r != HOLD)
    cursor = tl.load(cursors + r, valid, 0)
    n = cursor % (2 * L)
    phase = cursor & (2 * L)
    if COMMIT:
        count = tl.load(accepted + r, valid, 0) + 1 if WIDTH > 1 else 1
        next_cursor = tl.where(n + WIDTH > L, phase ^ (2 * L), phase + n) + count
        tl.store(cursors + r, next_cursor, valid)
    else:
        tl.store(out + i, cursor, i < N)




@triton.jit
def _replay(
    Q, Kp, Vp, A, B, Alog, Bias, State, Keys, Deltas, Gates, RawKeys, RawValues, Betas,
    Reqs, Cursors, Cu, Out,
    SQ: tl.constexpr, SK: tl.constexpr, SV: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    L: tl.constexpr, WIDTH: tl.constexpr, HOLD: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, VARLEN: tl.constexpr,
):
    """Sequential output-only path; faster than a padded MMA tile for short MTP windows."""
    iv, seq, hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(Reqs + seq).to(tl.int64)
    if VARLEN:
        start, end = tl.load(Cu + seq), tl.load(Cu + seq + 1)
    else:
        start, end = seq, seq + 1
    vv = iv * BV + tl.arange(0, BV)
    if req == HOLD:
        for t in range(start, end):
            tl.store(Out + (t * HV + hv) * V + vv, 0, vv < V)
        return
    if start == end:
        return

    kk = tl.arange(0, BK)
    ll = tl.arange(0, L)
    slot = req * HV + hv
    cursor = tl.load(Cursors + seq)
    n = cursor % (2 * L)
    base = (cursor // (2 * L)) * L
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)

    if n + WIDTH > L:
        for j in range(n):
            raw_k = tl.load(RawKeys + (slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
            raw_v = tl.load(RawValues + (slot * (2 * L) + base + j) * V + vv, vv < V, 0).to(tl.float32)
            g = tl.load(Gates + slot * (2 * L) + base + j).to(tl.float32)
            beta = tl.load(Betas + slot * (2 * L) + base + j).to(tl.float32)
            raw_k /= tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
            state *= tl.exp(g)
            d = beta * (raw_v - tl.sum(state * raw_k[:, None], 0))
            state += raw_k[:, None] * d[None, :]
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
        base, n = L - base, 0

    history = ll < n
    keys = tl.load(
        Keys + (slot * (2 * L) + base) * K + ll[:, None] * K + kk[None, :],
        history[:, None] & (kk[None, :] < K), 0,
    ).to(tl.float32)
    deltas = tl.load(
        Deltas + (slot * (2 * L) + base) * V + ll[:, None] * V + vv[None, :],
        history[:, None] & (vv[None, :] < V), 0,
    ).to(tl.float32)
    gates = tl.load(Gates + slot * (2 * L) + base + ll, history, 0).to(tl.float32)
    gate_prefix = tl.cumsum(gates, axis=0)
    total_g = tl.sum(gates, axis=0)
    h = hv // (HV // H)
    log_a, bias = tl.load(Alog + hv).to(tl.float32), tl.load(Bias + hv).to(tl.float32)

    for t in range(start, end):
        q = tl.load(Q + t * SQ + h * K + kk, kk < K, 0).to(tl.float32)
        raw_k = tl.load(Kp + t * SK + h * K + kk, kk < K, 0).to(tl.float32)
        v = tl.load(Vp + t * SV + hv * V + vv, vv < V, 0).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q) + 1.0e-6) * (K**-0.5)
        k = raw_k / tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
        x = tl.load(A + t * SA + hv).to(tl.float32) + bias
        g = -tl.exp(log_a) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        beta = tl.sigmoid(tl.load(B + t * SB + hv).to(tl.float32))
        total_g += g
        weights = tl.where(history, tl.exp(total_g - gate_prefix), 0.0)
        hk = tl.sum(keys * k[None, :], axis=1) * weights
        hq = tl.sum(keys * q[None, :], axis=1) * weights
        sk = tl.sum(state * k[:, None], axis=0) * tl.exp(total_g) + tl.sum(deltas * hk[:, None], axis=0)
        sq = tl.sum(state * q[:, None], axis=0) * tl.exp(total_g) + tl.sum(deltas * hq[:, None], axis=0)
        d = beta * (v - sk)
        tl.store(Out + (t * HV + hv) * V + vv, sq + d * tl.sum(k * q), vv < V)
        record = base + n
        tl.store(Deltas + (slot * (2 * L) + record) * V + vv, d, vv < V)
        tl.store(RawValues + (slot * (2 * L) + record) * V + vv, v, vv < V)
        if iv == 0:
            tl.store(Keys + (slot * (2 * L) + record) * K + kk, k, kk < K)
            tl.store(RawKeys + (slot * (2 * L) + record) * K + kk, raw_k, kk < K)
            tl.store(Gates + slot * (2 * L) + record, g)
            tl.store(Betas + slot * (2 * L) + record, beta)
        keys = tl.where((ll == n)[:, None], k[None, :], keys)
        deltas = tl.where((ll == n)[:, None], d[None, :], deltas)
        gate_prefix = tl.where(ll >= n, total_g, gate_prefix)
        history |= ll == n
        n += 1


@triton.jit
def _materialize(
    State,
    RawKeys,
    RawValues,
    Gates,
    Betas,
    Cursors,
    Reqs,
    SLOTS: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    HOLD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    iv, row, lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(Reqs + row).to(tl.int64)
    if req == HOLD:
        return
    cursor = tl.load(Cursors + req)
    n = cursor % (2 * L)
    base = (cursor // (2 * L)) * L
    if n == 0:
        return
    layer, head = lh // HV, lh % HV
    slot = (layer * SLOTS + req) * HV + head
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    for j in range(n):
        raw_k = tl.load(RawKeys + (slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
        raw_v = tl.load(RawValues + (slot * (2 * L) + base + j) * V + vv, vv < V, 0).to(tl.float32)
        g = tl.load(Gates + slot * (2 * L) + base + j).to(tl.float32)
        beta = tl.load(Betas + slot * (2 * L) + base + j).to(tl.float32)
        raw_k /= tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
        state *= tl.exp(g)
        d = beta * (raw_v - tl.sum(state * raw_k[:, None], 0))
        state += raw_k[:, None] * d[None, :]
    tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))


class ReplaySSMCache:
    """Request-owned scratch; canonical CPU/PD checkpoints never include it."""

    def __init__(self, state, capacity, verify_width, activation_dtype=torch.bfloat16):
        assert state.dtype == torch.float32, "deferred ReplaySSM requires FP32 SSM state"
        assert capacity >= max(16, verify_width) and capacity & (capacity - 1) == 0
        self.state = state
        self.capacity = capacity
        self.verify_width = verify_width
        layers, slots, hv, k, v = state.shape
        self.hold = slots - 1
        self.cursors = torch.zeros(slots, dtype=torch.int32, device=state.device)
        # Alternate halves on fold so CTAs cannot overwrite history another
        # V tile is still reading. Cursor packs phase (2*L) and count (0..L).
        self.keys = torch.empty((layers, slots, hv, 2 * capacity, k), dtype=torch.float32, device=state.device)
        self.deltas = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=torch.float32, device=state.device)
        self.gates = torch.empty((layers, slots, hv, 2 * capacity), dtype=torch.float32, device=state.device)
        self.raw_keys = torch.empty((layers, slots, hv, 2 * capacity, k), dtype=activation_dtype, device=state.device)
        self.raw_values = torch.empty(
            (layers, slots, hv, 2 * capacity, v), dtype=activation_dtype, device=state.device
        )
        self.betas = torch.empty((layers, slots, hv, 2 * capacity), dtype=torch.float32, device=state.device)

    def reset(self, req):
        self.cursors[req] = 0

    def positions(self, reqs):
        positions = torch.empty_like(reqs)
        _positions[(triton.cdiv(reqs.numel(), 256),)](
            self.cursors,
            reqs,
            positions,
            None,
            reqs.numel(),
            self.hold,
            self.capacity,
            self.verify_width,
            False,
            256,
        )
        return positions

    def commit(self, reqs, accepted=None):
        _positions[(triton.cdiv(reqs.numel(), 256),)](
            self.cursors,
            reqs,
            None,
            accepted,
            reqs.numel(),
            self.hold,
            self.capacity,
            self.verify_width,
            True,
            256,
        )

    def materialize(self, reqs):
        if reqs.numel() == 0:
            return
        layers, slots, hv, k, v = self.state.shape
        _materialize[(triton.cdiv(v, 32), reqs.numel(), layers * hv)](
            self.state,
            self.raw_keys,
            self.raw_values,
            self.gates,
            self.betas,
            self.cursors,
            reqs,
            slots,
            hv,
            k,
            v,
            self.capacity,
            self.hold,
            triton.next_power_of_2(k),
            32,
        )
        self.cursors[reqs] = 0

    def forward(self, layer, q, k, v, a, b, a_log, bias, reqs, positions, cu_seqlens=None, run_config=None):
        hv, kd, vd = self.state.shape[-3:]
        axis = 1 if cu_seqlens is not None else 0
        out = torch.empty_like(v)
        config = run_config or {"BV": 64, "num_warps": 2}
        bv = config["BV"]
        _replay[(triton.cdiv(vd, bv), reqs.numel(), hv)](
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            self.state[layer],
            self.keys[layer],
            self.deltas[layer],
            self.gates[layer],
            self.raw_keys[layer],
            self.raw_values[layer],
            self.betas[layer],
            reqs,
            positions,
            cu_seqlens,
            out,
            q.stride(axis),
            k.stride(axis),
            v.stride(axis),
            a.stride(0),
            b.stride(0),
            q.shape[-2],
            hv,
            kd,
            vd,
            self.capacity,
            self.verify_width,
            self.hold,
            triton.next_power_of_2(kd),
            bv,
            cu_seqlens is not None,
            num_warps=config["num_warps"],
        )
        return out
