# SPDX-License-Identifier: Apache-2.0
"""GDN/KDA ReplaySSM in LightLLM's [K, V] state layout.

The checkpoint contains only folded tokens. Derived records reconstruct verify
outputs, while raw inputs replay the accepted suffix into the checkpoint. Verify
may overwrite the uncommitted suffix; only acceptance advances the cursor.
FP32 accumulation is rounded to the checkpoint dtype only on fold/materialize;
BF16 replay therefore differs from native per-token BF16 state rounding.
Algorithm reference: https://dao-lab.ai/blog/2026/replayssm/
"""

import torch
import triton
import triton.language as tl

from .ssm_autotune import configure_cache


@triton.jit
def _accept_updates(
    cursors,
    reqs,
    accepted,
    N: tl.constexpr,
    HOLD: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(reqs + i, i < N, HOLD)
    valid = (i < N) & (r != HOLD)
    cursor = tl.load(cursors + r, valid, 0)
    count = tl.load(accepted + r, valid, 0) + 1 if WIDTH > 1 else 1
    tl.store(cursors + r, cursor + count, valid)


@triton.jit
def _prepare_fold(
    Reqs,
    Cursors,
    Positions,
    ForwardCursors,
    Cu,
    Active,
    N: tl.constexpr,
    HOLD: tl.constexpr,
    L: tl.constexpr,
    VARLEN: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
):
    seq = tl.arange(0, BLOCK)
    req = tl.load(Reqs + seq, seq < N, HOLD)
    valid = (seq < N) & (req != HOLD)
    cursor = tl.load(Cursors + req, valid, 0)
    tl.store(Positions + seq, cursor, seq < N)
    if VARLEN:
        start = tl.load(Cu + seq, seq < N, 0)
        end = tl.load(Cu + seq + 1, seq < N, 0)
        length = end - start
    else:
        length = tl.full((BLOCK,), 1, tl.int32)
    overflow = cursor % (2 * L) + length > L
    tl.store(Cursors + req, (cursor & (2 * L)) ^ (2 * L), valid & overflow)
    needs_fold = valid & (length > 0) & overflow
    offset = tl.cumsum(needs_fold.to(tl.int32), axis=0) - 1
    tl.store(Active + offset, seq, needs_fold)
    count = tl.sum(needs_fold.to(tl.int32), axis=0)
    live = tl.sum((valid & (length > 0)).to(tl.int32), axis=0)
    split = (count > 0) & ((live <= 12) | (2 * count <= live))
    if N >= 256 and L == 8 and WIDTH == 4:
        short = tl.sum((valid & (length > 0) & (length < WIDTH)).to(tl.int32), axis=0)
        split = split & ((live < 256) | (short > 0))
    tl.store(ForwardCursors + req, tl.where(overflow & split, (cursor & (2 * L)) ^ (2 * L), cursor), valid)
    tl.store(Active + N, tl.where(split, count, 0))


@triton.jit
def _replay_update(state, raw_k, raw_v, g, beta, KDA: tl.constexpr):
    raw_k /= tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
    decay = tl.exp(g)
    if KDA:
        state *= decay[:, None]
        d = beta * (raw_v - tl.sum(state * raw_k[:, None], 0))
        state += raw_k[:, None] * d[None, :]
    else:
        sk = tl.sum(state * raw_k[:, None], 0)
        d = beta * (raw_v - decay * sk)
        state = state * decay + raw_k[:, None] * d[None, :]
    return state


@triton.jit
def _fold_history(
    State,
    RawKeys,
    RawValues,
    Gates,
    Betas,
    Reqs,
    Cursors,
    Active,
    N: tl.constexpr,
    LAYERS: tl.constexpr,
    HV: tl.constexpr,
    KH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SS: tl.constexpr,
    RS: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    KDA: tl.constexpr,
):
    count = tl.load(Active + N)
    work = tl.program_id(0)
    tiles: tl.constexpr = (V + BV - 1) // BV
    total = count * LAYERS * HV * tiles
    while work < total:
        iv = work % tiles
        slot_work = work // tiles
        seq = tl.load(Active + slot_work % count)
        layer_hv = slot_work // count
        layer, hv = (layer_hv // HV).to(tl.int64), layer_hv % HV
        req = tl.load(Reqs + seq).to(tl.int64)
        cursor = tl.load(Cursors + seq)
        n = cursor % (2 * L)
        base = (cursor // (2 * L)) * L
        kk = tl.arange(0, BK)
        vv = iv * BV + tl.arange(0, BV)
        slot = req * HV + hv
        key_slot = req * KH + hv // (HV // KH)
        sp = State + layer * SS + slot * K * V + kk[:, None] * V + vv[None, :]
        state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
        for j in range(n):
            raw_k = tl.load(RawKeys + layer * RS + (key_slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
            raw_v = tl.load(RawValues + layer * VS + (slot * (2 * L) + base + j) * V + vv, vv < V, 0).to(tl.float32)
            if KDA:
                g = tl.load(Gates + layer * GS + (slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
            else:
                g = tl.load(Gates + layer * GS + slot * (2 * L) + base + j).to(tl.float32)
            beta = tl.load(Betas + layer * BS + slot * (2 * L) + base + j).to(tl.float32)
            state = _replay_update(state, raw_k, raw_v, g, beta, KDA)
        # Start the new history from the same rounded checkpoint future calls load.
        state = state.to(State.dtype.element_ty).to(tl.float32)
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
        work += tl.num_programs(0)


@triton.jit
def _replay(
    Q,
    Kp,
    Vp,
    A,
    B,
    Alog,
    Bias,
    State,
    Keys,
    Deltas,
    Gates,
    RawKeys,
    RawValues,
    Betas,
    Reqs,
    Cursors,
    Cu,
    Out,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    KH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    HOLD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
    PRECOMPUTE_STATE: tl.constexpr,
    BW: tl.constexpr,
    FUSED_FOLD: tl.constexpr,
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
):
    """Output reconstruction with an exact raw-input recurrence at checkpoint folds."""
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
    key_slot = req * KH + hv // (HV // KH)
    cursor = tl.load(Cursors + req)
    n = cursor % (2 * L)
    base = (cursor // (2 * L)) * L
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)

    if FUSED_FOLD and n + end - start > L:
        for j in range(n):
            raw_k = tl.load(RawKeys + (key_slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
            raw_v = tl.load(RawValues + (slot * (2 * L) + base + j) * V + vv, vv < V, 0).to(tl.float32)
            if KDA:
                g = tl.load(Gates + (slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(tl.float32)
            else:
                g = tl.load(Gates + slot * (2 * L) + base + j).to(tl.float32)
            beta = tl.load(Betas + slot * (2 * L) + base + j).to(tl.float32)
            state = _replay_update(state, raw_k, raw_v, g, beta, KDA)
        # Start the new history from the same rounded checkpoint future calls load.
        state = state.to(State.dtype.element_ty).to(tl.float32)
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
        base, n = L - base, 0

    if PRECOMPUTE_STATE:
        projection_dtype: tl.constexpr = (
            tl.bfloat16 if State.dtype.element_ty == tl.bfloat16 and Q.dtype.element_ty == tl.bfloat16 else tl.float32
        )
        projection_precision: tl.constexpr = "tf32" if Q.dtype.element_ty == tl.bfloat16 else "tf32x3"
        tt = tl.arange(0, max(16, 2 * BW))
        tokens = start + tt % BW
        valid = (tt < 2 * BW) & (tokens < end)
        h = hv // (HV // H)
        q_rows = tl.load(
            Q + tokens[:, None] * SQ + h * K + kk[None, :],
            (valid & (tt < BW))[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        k_rows = tl.load(
            Kp + tokens[:, None] * SK + h * K + kk[None, :],
            (valid & (tt >= BW))[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        qk = q_rows + k_rows
        qk *= tl.rsqrt(tl.sum(qk * qk, axis=1) + 1.0e-6)[:, None]
        qk *= tl.where(tt < BW, K ** -0.5, 1.0)[:, None]
        if KDA:
            history = ll < n
            gates = tl.load(
                Gates + (slot * (2 * L) + base + ll[:, None]) * K + kk[None, :],
                history[:, None] & (kk[None, :] < K),
                0,
            ).to(tl.float32)
            total_g = tl.sum(gates, axis=0)
            wi = tl.arange(0, BW)
            gate_inputs = tl.load(
                A + (start + wi[:, None]) * SA + hv * K + kk[None, :],
                (start + wi[:, None] < end) & (kk[None, :] < K),
                0,
            ).to(tl.float32)
            gate_inputs += tl.load(Bias + hv * K + kk, kk < K, 0).to(tl.float32)[None, :]
            fresh_gates = LOWER * tl.sigmoid(tl.exp(tl.load(Alog + hv)) * gate_inputs)
            fresh_gates = tl.where((start + wi < end)[:, None], fresh_gates, 0)
            query_gates = total_g[None, :] + tl.gather(
                tl.cumsum(fresh_gates, axis=0),
                tl.broadcast_to((tt % BW)[:, None], qk.shape),
                0,
            )
            # All exponents are forward decay. No inverse prefix scaling can
            # overflow when a long history contains strongly negative gates.
            projections = tl.dot(
                (qk * tl.exp(query_gates)).to(projection_dtype),
                state.to(projection_dtype),
                input_precision=projection_precision,
            )
        else:
            projections = tl.dot(
                qk.to(projection_dtype), state.to(projection_dtype), input_precision=projection_precision
            )

    if PRECOMPUTE_STATE and not KDA:
        # Project the whole verify window against both accepted and new keys.
        # Only the small corrected-value solve below depends on token order.
        jj = tl.arange(0, triton.next_power_of_2(max(16, L + BW)))
        fresh_tokens = start + jj - L
        old = (jj < L) & (jj < n)
        fresh = (jj >= L) & (jj < L + end - start)
        cached_keys = tl.load(
            Keys + (key_slot * (2 * L) + base) * K + jj[:, None] * K + kk[None, :],
            old[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        raw_keys = tl.load(
            Kp + fresh_tokens[:, None] * SK + h * K + kk[None, :],
            fresh[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        fresh_keys = raw_keys * tl.rsqrt(tl.sum(raw_keys * raw_keys, axis=1) + 1.0e-6)[:, None]
        all_keys = cached_keys + fresh_keys
        values = tl.load(
            Vp + fresh_tokens[:, None] * SV + hv * V + vv[None, :],
            fresh[:, None] & (vv[None, :] < V),
            0,
        ).to(tl.float32)
        log_a, bias = tl.load(Alog + hv).to(tl.float32), tl.load(Bias + hv).to(tl.float32)
        x = tl.load(A + fresh_tokens * SA + hv, fresh, 0).to(tl.float32) + bias
        fresh_gates = -tl.exp(log_a) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        gates = tl.load(Gates + slot * (2 * L) + base + jj, old, 0).to(tl.float32)
        gates += tl.where(fresh, fresh_gates, 0.0)
        prefix = tl.cumsum(gates, axis=0)
        query_gates = tl.sum(tl.where(jj[None, :] <= L + (tt % BW)[:, None], gates[None, :], 0), 1)
        coefficients = tl.dot(
            qk.to(projection_dtype), tl.trans(all_keys).to(projection_dtype), input_precision=projection_precision
        )
        causal = old[None, :] | (fresh[None, :] & (jj[None, :] <= L + (tt % BW)[:, None]))
        coefficients *= tl.where(causal, tl.exp(tl.minimum(query_gates[:, None] - prefix[None, :], 0.0)), 0.0)
        projections *= tl.exp(query_gates)[:, None]
        deltas = tl.load(
            Deltas + (slot * (2 * L) + base) * V + jj[:, None] * V + vv[None, :],
            old[:, None] & (vv[None, :] < V),
            0,
        ).to(tl.float32)
        betas = tl.sigmoid(tl.load(B + fresh_tokens * SB + hv, fresh, 0).to(tl.float32))
        for t in range(end - start):
            checkpoint_sk = tl.sum(tl.where((tt == BW + t)[:, None], projections, 0), 0)
            weights = tl.sum(tl.where((tt == BW + t)[:, None], coefficients, 0), 0)
            sk = checkpoint_sk + tl.sum(deltas * weights[:, None], 0)
            value = tl.sum(tl.where((jj == L + t)[:, None], values, 0), 0)
            beta = tl.sum(tl.where(jj == L + t, betas, 0))
            delta = beta * (value - sk)
            deltas = tl.where((jj == L + t)[:, None], delta[None, :], deltas)
        outputs = projections + tl.dot(
            coefficients.to(projection_dtype), deltas.to(projection_dtype), input_precision=projection_precision
        )
        tl.store(
            Out + (tokens[:, None] * HV + hv) * V + vv[None, :],
            outputs,
            (valid & (tt < BW))[:, None] & (vv[None, :] < V),
        )
        records = base + n + jj - L
        tl.store(
            Deltas + (slot * (2 * L) + records[:, None]) * V + vv[None, :],
            deltas,
            fresh[:, None] & (vv[None, :] < V),
        )
        tl.store(
            RawValues + (slot * (2 * L) + records[:, None]) * V + vv[None, :],
            values,
            fresh[:, None] & (vv[None, :] < V),
        )
        if iv == 0:
            if hv % (HV // KH) == 0:
                tl.store(
                    Keys + (key_slot * (2 * L) + records[:, None]) * K + kk[None, :],
                    fresh_keys,
                    fresh[:, None] & (kk[None, :] < K),
                )
                tl.store(
                    RawKeys + (key_slot * (2 * L) + records[:, None]) * K + kk[None, :],
                    raw_keys,
                    fresh[:, None] & (kk[None, :] < K),
                )
            tl.store(Gates + slot * (2 * L) + records, fresh_gates, fresh)
            tl.store(Betas + slot * (2 * L) + records, betas, fresh)
        return

    if not PRECOMPUTE_STATE or KDA:
        history = ll < n
        keys = tl.load(
            Keys + (key_slot * (2 * L) + base) * K + ll[:, None] * K + kk[None, :],
            history[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        deltas = tl.load(
            Deltas + (slot * (2 * L) + base) * V + ll[:, None] * V + vv[None, :],
            history[:, None] & (vv[None, :] < V),
            0,
        ).to(tl.float32)
        if KDA:
            gates = tl.load(
                Gates + (slot * (2 * L) + base + ll[:, None]) * K + kk[None, :],
                history[:, None] & (kk[None, :] < K),
                0,
            ).to(tl.float32)
        else:
            gates = tl.load(Gates + slot * (2 * L) + base + ll, history, 0).to(tl.float32)
        gate_prefix = tl.cumsum(gates, axis=0)
        total_g = tl.sum(gates, axis=0)
        if KDA:
            keys *= tl.where(history[:, None], tl.exp(total_g[None, :] - gate_prefix), 0.0)
        h = hv // (HV // H)
        log_a = tl.load(Alog + hv).to(tl.float32)
        bias = tl.load(Bias + hv * K + kk, kk < K, 0).to(tl.float32) if KDA else tl.load(Bias + hv).to(tl.float32)

        for t in range(start, end):
            q = tl.load(Q + t * SQ + h * K + kk, kk < K, 0).to(tl.float32)
            raw_k = tl.load(Kp + t * SK + h * K + kk, kk < K, 0).to(tl.float32)
            v = tl.load(Vp + t * SV + hv * V + vv, vv < V, 0).to(tl.float32)
            q = q / tl.sqrt(tl.sum(q * q) + 1.0e-6) * (K ** -0.5)
            k = raw_k / tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
            if KDA:
                x = tl.load(A + t * SA + hv * K + kk, kk < K, 0).to(tl.float32) + bias
                g = LOWER * tl.sigmoid(tl.exp(log_a) * x)
            else:
                x = tl.load(A + t * SA + hv).to(tl.float32) + bias
                g = -tl.exp(log_a) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
            beta = tl.sigmoid(tl.load(B + t * SB + hv).to(tl.float32))
            total_g += g
            if KDA:
                # Advance the decayed keys once per token, rather than
                # evaluating L independent vector exponentials each time.
                keys *= tl.exp(g)[None, :]
                hk = tl.sum(keys * k[None, :], axis=1)
                hq = tl.sum(keys * q[None, :], axis=1)
                if PRECOMPUTE_STATE:
                    checkpoint_sk = tl.sum(tl.where((tt == BW + t - start)[:, None], projections, 0), 0)
                    checkpoint_sq = tl.sum(tl.where((tt == t - start)[:, None], projections, 0), 0)
                else:
                    checkpoint_sk = tl.sum(state * (k * tl.exp(total_g))[:, None], axis=0)
                    checkpoint_sq = tl.sum(state * (q * tl.exp(total_g))[:, None], axis=0)
                sk = checkpoint_sk + tl.sum(deltas * hk[:, None], axis=0)
                sq = checkpoint_sq + tl.sum(deltas * hq[:, None], axis=0)
            else:
                weights = tl.where(history, tl.exp(total_g - gate_prefix), 0.0)
                hk = tl.sum(keys * k[None, :], axis=1) * weights
                hq = tl.sum(keys * q[None, :], axis=1) * weights
                checkpoint_sk = tl.sum(state * k[:, None], axis=0)
                checkpoint_sq = tl.sum(state * q[:, None], axis=0)
                sk = checkpoint_sk * tl.exp(total_g) + tl.sum(deltas * hk[:, None], axis=0)
                sq = checkpoint_sq * tl.exp(total_g) + tl.sum(deltas * hq[:, None], axis=0)
            d = beta * (v - sk)
            tl.store(Out + (t * HV + hv) * V + vv, sq + d * tl.sum(k * q), vv < V)
            record = base + n
            tl.store(Deltas + (slot * (2 * L) + record) * V + vv, d, vv < V)
            tl.store(RawValues + (slot * (2 * L) + record) * V + vv, v, vv < V)
            if iv == 0:
                # Only the group owner writes shared K. Every CTA computes this
                # round's K locally; readers only consume earlier accepted records.
                if hv % (HV // KH) == 0:
                    tl.store(Keys + (key_slot * (2 * L) + record) * K + kk, k, kk < K)
                    tl.store(RawKeys + (key_slot * (2 * L) + record) * K + kk, raw_k, kk < K)
                if KDA:
                    tl.store(Gates + (slot * (2 * L) + record) * K + kk, g, kk < K)
                else:
                    tl.store(Gates + slot * (2 * L) + record, g)
                tl.store(Betas + slot * (2 * L) + record, beta)
            keys = tl.where((ll == n)[:, None], k[None, :], keys)
            deltas = tl.where((ll == n)[:, None], d[None, :], deltas)
            if not KDA:
                gate_prefix = tl.where(ll >= n, total_g, gate_prefix)
            history |= ll == n
            n += 1


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
    layer, head = lh // HV, lh % HV
    slot = (layer * SLOTS + req) * HV + head
    key_slot = (layer * SLOTS + req) * KH + head // (HV // KH)
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
    out_slot = layer * HV + head if SNAPSHOT else slot
    out = Output + out_slot * K * V + kk[:, None] * V + vv[None, :]
    tl.store(out, state, (kk[:, None] < K) & (vv[None, :] < V))


class ReplaySSMCache:
    """Request-owned scratch; canonical CPU/PD checkpoints never include it."""

    def __init__(
        self,
        state,
        capacity,
        verify_width,
        activation_dtype=torch.bfloat16,
        *,
        num_key_heads=None,
        kda=False,
        lower_bound=-5.0,
        projection_mode="inline",
        run_config=None,
    ):
        assert state.dtype in (torch.float32, torch.bfloat16), "ReplaySSM requires FP32 or BF16 SSM state"
        assert verify_width > 0 and capacity >= max(4, verify_width) and capacity & (capacity - 1) == 0
        assert projection_mode in ("inline", "precompute")
        self.state = state
        self.kda = kda
        self.lower_bound = lower_bound
        self.fold_programs = (
            8 * torch.cuda.get_device_properties(state.device).multi_processor_count if state.is_cuda else 0
        )
        self.capacity = capacity
        self.verify_width = verify_width
        self.projection_mode = projection_mode
        self.run_config = run_config
        self._config_is_fixed = run_config is not None
        self.batch_configs = {}
        layers, slots, hv, k, v = state.shape
        # Omission preserves the redundant layout for existing callers and A/B.
        self.num_key_heads = hv if num_key_heads is None else num_key_heads
        assert self.num_key_heads > 0 and hv % self.num_key_heads == 0
        self.hold = slots - 1
        self.cursors = torch.zeros(slots, dtype=torch.int32, device=state.device)
        self.forward_cursors = torch.zeros_like(self.cursors)
        # Alternate halves on fold so CTAs cannot overwrite history another
        # V tile is still reading. Cursor packs phase (2*L) and count (0..L).
        history_dtype = (
            activation_dtype
            if verify_width > 1 and projection_mode == "precompute" and capacity <= 8
            else torch.float32
        )
        key_shape = (layers, slots, self.num_key_heads, 2 * capacity, k)
        self.keys = torch.empty(key_shape, dtype=history_dtype, device=state.device)
        self.deltas = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=history_dtype, device=state.device)
        self.gates = torch.empty(
            (layers, slots, hv, 2 * capacity, k) if kda else (layers, slots, hv, 2 * capacity),
            dtype=torch.float32,
            device=state.device,
        )
        self.raw_keys = torch.empty(key_shape, dtype=activation_dtype, device=state.device)
        self.raw_values = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=activation_dtype, device=state.device)
        self.betas = torch.empty((layers, slots, hv, 2 * capacity), dtype=torch.float32, device=state.device)

    def clear_history(self, req):
        """Invalidate history for a reused/restored request; leave its SSM state intact."""
        if type(req) is int:
            # Scalar assignment copies a CPU tensor into the CUDA slot.
            self.cursors[req].zero_()
        else:
            self.cursors[req] = 0

    def prepare_decode(self, reqs, cu_seqlens=None):
        """Snapshot history for all layers and prepare the base for accepted updates."""
        positions = torch.empty_like(reqs)
        layers, _, hv, kd, vd = self.state.shape
        active = torch.empty((reqs.numel() + 1,), dtype=torch.int32, device=reqs.device)
        _prepare_fold[(1,)](
            reqs,
            self.cursors,
            positions,
            self.forward_cursors,
            cu_seqlens,
            active,
            reqs.numel(),
            self.hold,
            self.capacity,
            cu_seqlens is not None,
            triton.next_power_of_2(reqs.numel()),
            self.verify_width,
            num_warps=4,
        )
        programs = min(self.fold_programs, triton.cdiv(vd, 32) * reqs.numel() * layers * hv)
        _fold_history[(programs,)](
            self.state,
            self.raw_keys,
            self.raw_values,
            self.gates,
            self.betas,
            reqs,
            positions,
            active,
            reqs.numel(),
            layers,
            hv,
            self.num_key_heads,
            kd,
            vd,
            self.capacity,
            triton.next_power_of_2(kd),
            32,
            self.state.stride(0),
            self.raw_keys.stride(0),
            self.raw_values.stride(0),
            self.gates.stride(0),
            self.betas.stride(0),
            self.kda,
            num_warps=2,
            num_stages=1,
        )
        return positions

    def accept_updates(self, reqs, accepted=None):
        """Accept this round's prefix; accepted contains per-request last accepted indexes."""
        _accept_updates[(triton.cdiv(reqs.numel(), 256),)](
            self.cursors,
            reqs,
            accepted,
            reqs.numel(),
            self.hold,
            self.verify_width,
            256,
        )

    def merge_accepted_updates(self, reqs):
        """Fold accepted history into SSM state and clear its history positions."""
        if reqs.numel() == 0:
            return
        self._materialize_accepted_state(reqs, self.state, snapshot=False)
        self.cursors.index_fill_(0, reqs.long(), 0)

    def _materialize_accepted_state(self, reqs, output, snapshot):
        layers, slots, hv, k, v = self.state.shape
        config = self.run_config or {"BV": 32, "num_warps": 4}
        bv = config["BV"]
        _materialize[(triton.cdiv(v, bv), reqs.numel(), layers * hv)](
            self.state,
            output,
            self.raw_keys,
            self.raw_values,
            self.gates,
            self.betas,
            self.cursors,
            reqs,
            slots,
            hv,
            self.num_key_heads,
            k,
            v,
            self.capacity,
            self.hold,
            triton.next_power_of_2(k),
            bv,
            snapshot,
            self.kda,
            num_warps=config["num_warps"],
            num_stages=config.get("num_stages", 3),
        )

    def snapshot_accepted_state(self, req_idx):
        """Return a canonical checkpoint without changing the active state or history."""
        layers, _, hv, k, v = self.state.shape
        output = torch.empty((layers, hv, k, v), dtype=self.state.dtype, device=self.state.device)
        reqs = torch.tensor([req_idx], dtype=torch.int32, device=self.state.device)
        self._materialize_accepted_state(reqs, output, snapshot=True)
        return output

    def get_run_config(self, q, k, v, a, b, a_log, bias, cu_seqlens=None):
        return configure_cache(self, "replay", q, k, v, a, b, a_log, bias, cu_seqlens)

    def forward(self, layer, q, k, v, a, b, a_log, bias, reqs, cu_seqlens=None, run_config=None):
        config = run_config or self.get_run_config(q, k, v, a, b, a_log, bias, cu_seqlens)
        hv, kd, vd = self.state.shape[-3:]
        assert self.num_key_heads in (q.shape[-2], hv)
        axis = 1 if cu_seqlens is not None else 0
        out = torch.empty_like(v)
        bv = config["BV"]
        precompute_state = self.verify_width > 1 and bv >= 16 and self.projection_mode == "precompute"
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
            self.forward_cursors,
            cu_seqlens,
            out,
            q.stride(axis),
            k.stride(axis),
            v.stride(axis),
            a.stride(0),
            b.stride(0),
            q.shape[-2],
            hv,
            self.num_key_heads,
            kd,
            vd,
            self.capacity,
            self.hold,
            triton.next_power_of_2(kd),
            bv,
            cu_seqlens is not None,
            precompute_state,
            triton.next_power_of_2(self.verify_width),
            reqs.numel() > 12,
            self.kda,
            self.lower_bound,
            num_warps=config["num_warps"],
            num_stages=config.get("num_stages", 3),
        )
        return out
