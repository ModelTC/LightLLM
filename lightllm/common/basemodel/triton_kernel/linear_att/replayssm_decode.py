# SPDX-License-Identifier: Apache-2.0
"""Double-buffered ReplaySSM decode kernels used by KDA and GDN."""

import triton
import triton.language as tl


@triton.jit
def _arrive_value_tiles(SyncMark, seq, hv, HV: tl.constexpr, KH: tl.constexpr, NV: tl.constexpr):
    """Legacy one-ring fallback barrier kept for old standalone fixtures."""

    group: tl.constexpr = HV // KH
    key_head = hv // group
    idx = seq * KH + key_head
    arrived = tl.atomic_add(SyncMark + idx, 1, sem="acq_rel", scope="gpu")
    if arrived == NV * group - 1:
        tl.atomic_xchg(SyncMark + idx, 0, sem="release", scope="gpu")
    else:
        seen = arrived + 1
        while seen != 0:
            seen = tl.atomic_add(SyncMark + idx, 0, sem="acquire", scope="gpu")


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
def _decode_replay(
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
    SyncMark,
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
    PRECOMPUTE_PROJECTIONS: tl.constexpr,
    BW: tl.constexpr,
    FUSED_FOLD: tl.constexpr,
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
    PARALLEL_SOLVE: tl.constexpr,
    BQK: tl.constexpr,
    BR: tl.constexpr,
    HIST: tl.constexpr,
    DOUBLE_BUFFER: tl.constexpr,
    NV: tl.constexpr,
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
    if DOUBLE_BUFFER:
        n = cursor % HIST
        base = (cursor // HIST) * L
    else:
        n = cursor
        base = 0
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)

    if FUSED_FOLD and n + end - start > L:
        for j in range(n):
            raw_k = tl.load(RawKeys + (key_slot * HIST + base + j) * K + kk, kk < K, 0).to(tl.float32)
            raw_v = tl.load(RawValues + (slot * HIST + base + j) * V + vv, vv < V, 0).to(tl.float32)
            if KDA:
                g = tl.load(Gates + (slot * HIST + base + j) * K + kk, kk < K, 0).to(tl.float32)
            else:
                g = tl.load(Gates + slot * HIST + base + j).to(tl.float32)
            beta = tl.load(Betas + slot * HIST + base + j).to(tl.float32)
            state = _replay_update(state, raw_k, raw_v, g, beta, KDA)
        # Start the new history from the same rounded checkpoint future calls load.
        state = state.to(State.dtype.element_ty).to(tl.float32)
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
        if DOUBLE_BUFFER:
            base, n = L - base, 0
        else:
            base, n = 0, 0
            _arrive_value_tiles(SyncMark, seq, hv, HV, KH, NV)

    if PRECOMPUTE_PROJECTIONS:
        projection_dtype: tl.constexpr = (
            tl.bfloat16 if State.dtype.element_ty == tl.bfloat16 and Q.dtype.element_ty == tl.bfloat16 else tl.float32
        )
        projection_precision: tl.constexpr = "tf32" if Q.dtype.element_ty == tl.bfloat16 else "tf32x3"
        tt = tl.arange(0, BQK)
        tokens = start + tt % BW
        valid = (tt < 2 * BW) & (tokens < end)
        safe_tokens = tl.where(valid, tokens, start)
        h = hv // (HV // H)
        q_rows = tl.load(
            Q + safe_tokens[:, None] * SQ + h * K + kk[None, :],
            (valid & (tt < BW))[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        k_rows = tl.load(
            Kp + safe_tokens[:, None] * SK + h * K + kk[None, :],
            (valid & (tt >= BW))[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        row_valid = valid[:, None]
        qk = tl.where(row_valid, q_rows + k_rows, 0)
        if PARALLEL_SOLVE:
            qk /= tl.sqrt(tl.sum(qk * qk, axis=1) + 1.0e-6)[:, None]
        else:
            qk *= tl.rsqrt(tl.sum(qk * qk, axis=1) + 1.0e-6)[:, None]
        qk *= tl.where(tt < BW, K ** -0.5, 1.0)[:, None]
        qk = tl.where(row_valid, qk, 0)
        if KDA:
            history = ll < n
            gates = tl.load(
                Gates + (slot * HIST + base + ll[:, None]) * K + kk[None, :],
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
                tl.where(row_valid, qk * tl.exp(query_gates), 0).to(projection_dtype),
                state.to(projection_dtype),
                input_precision=projection_precision,
            )
        else:
            projections = tl.dot(
                qk.to(projection_dtype), state.to(projection_dtype), input_precision=projection_precision
            )

    if PRECOMPUTE_PROJECTIONS and not KDA:
        # Project the whole verify window against both accepted and new keys.
        # Only the small corrected-value solve below depends on token order.
        jj = tl.arange(0, BR)
        fresh_tokens = start + jj - L
        old = (jj < L) & (jj < n)
        fresh = (jj >= L) & (jj < L + end - start)
        cached_keys = tl.load(
            Keys + (key_slot * HIST + base) * K + jj[:, None] * K + kk[None, :],
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
        gates = tl.load(Gates + slot * HIST + base + jj, old, 0).to(tl.float32)
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
            Deltas + (slot * HIST + base) * V + jj[:, None] * V + vv[None, :],
            old[:, None] & (vv[None, :] < V),
            0,
        ).to(tl.float32)
        betas = tl.sigmoid(tl.load(B + fresh_tokens * SB + hv, fresh, 0).to(tl.float32))
        if PARALLEL_SOLVE:
            projected_history = projections + tl.dot(
                coefficients.to(projection_dtype), deltas.to(projection_dtype), input_precision=projection_precision
            )
            wi = tl.arange(0, BQK)
            active = (wi < BW) & (start + wi < end)
            key_rows = tl.gather(
                projected_history, tl.broadcast_to((BW + wi % BW)[:, None], projected_history.shape), 0
            )
            fresh_values = tl.gather(values, tl.broadcast_to((L + wi % BW)[:, None], (BQK, BV)), 0)
            beta_rows = tl.sigmoid(tl.load(B + (start + wi) * SB + hv, active, 0).to(tl.float32))
            rhs = tl.where(active[:, None], beta_rows[:, None] * (fresh_values - key_rows), 0.0)
            cross = tl.gather(coefficients, tl.broadcast_to((BW + wi % BW)[:, None], coefficients.shape), 0)
            cross = tl.gather(cross, tl.broadcast_to((L + wi % BW)[None, :], (BQK, BQK)), 1)
            lower = active[:, None] & active[None, :] & (wi[:, None] > wi[None, :])
            power = tl.where(lower, -beta_rows[:, None] * cross, 0.0)
            inverse = (wi[:, None] == wi[None, :]).to(tl.float32)
            # The strictly lower-triangular coupling is nilpotent. Repeated
            # squaring builds (I - power)^-1 in log2(BW) matrix steps; capacity
            # is at most 64, so six steps cover every supported verify width.
            for solve in tl.static_range(6):
                if (1 << solve) < BW:
                    inverse += tl.dot(power, inverse, input_precision="tf32x3")
                    power = tl.dot(power, power, input_precision="tf32x3")
            solved = tl.dot(
                inverse.to(projection_dtype), rhs.to(projection_dtype), input_precision=projection_precision
            )
            mapped = tl.gather(solved, tl.broadcast_to(((jj - L) % BW)[:, None], deltas.shape), 0)
            deltas = tl.where(fresh[:, None], mapped, deltas)
        else:
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
            Deltas + (slot * HIST + records[:, None]) * V + vv[None, :],
            deltas,
            fresh[:, None] & (vv[None, :] < V),
        )
        tl.store(
            RawValues + (slot * HIST + records[:, None]) * V + vv[None, :],
            values,
            fresh[:, None] & (vv[None, :] < V),
        )
        if iv == 0:
            if hv % (HV // KH) == 0:
                tl.store(
                    Keys + (key_slot * HIST + records[:, None]) * K + kk[None, :],
                    fresh_keys,
                    fresh[:, None] & (kk[None, :] < K),
                )
                tl.store(
                    RawKeys + (key_slot * HIST + records[:, None]) * K + kk[None, :],
                    raw_keys,
                    fresh[:, None] & (kk[None, :] < K),
                )
            tl.store(Gates + slot * HIST + records, fresh_gates, fresh)
            tl.store(Betas + slot * HIST + records, betas, fresh)
        return

    if not PRECOMPUTE_PROJECTIONS or KDA:
        history = ll < n
        keys = tl.load(
            Keys + (key_slot * HIST + base) * K + ll[:, None] * K + kk[None, :],
            history[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        deltas = tl.load(
            Deltas + (slot * HIST + base) * V + ll[:, None] * V + vv[None, :],
            history[:, None] & (vv[None, :] < V),
            0,
        ).to(tl.float32)
        if KDA:
            gates = tl.load(
                Gates + (slot * HIST + base + ll[:, None]) * K + kk[None, :],
                history[:, None] & (kk[None, :] < K),
                0,
            ).to(tl.float32)
        else:
            gates = tl.load(Gates + slot * HIST + base + ll, history, 0).to(tl.float32)
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
                if PRECOMPUTE_PROJECTIONS:
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
            tl.store(Deltas + (slot * HIST + record) * V + vv, d, vv < V)
            tl.store(RawValues + (slot * HIST + record) * V + vv, v, vv < V)
            if iv == 0:
                # Only the group owner writes shared K. Every CTA computes this
                # round's K locally; readers only consume earlier accepted records.
                if hv % (HV // KH) == 0:
                    tl.store(Keys + (key_slot * HIST + record) * K + kk, k, kk < K)
                    tl.store(RawKeys + (key_slot * HIST + record) * K + kk, raw_k, kk < K)
                if KDA:
                    tl.store(Gates + (slot * HIST + record) * K + kk, g, kk < K)
                else:
                    tl.store(Gates + slot * HIST + record, g)
                tl.store(Betas + slot * HIST + record, beta)
            keys = tl.where((ll == n)[:, None], k[None, :], keys)
            deltas = tl.where((ll == n)[:, None], d[None, :], deltas)
            if not KDA:
                gate_prefix = tl.where(ll >= n, total_g, gate_prefix)
            history |= ll == n
            n += 1
