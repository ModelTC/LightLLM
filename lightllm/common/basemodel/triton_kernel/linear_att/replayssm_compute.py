# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM projection and recurrent decode kernels."""

import triton
import triton.language as tl


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
    PRECOMPUTE_PROJECTIONS: tl.constexpr,
    BW: tl.constexpr,
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
    BQK: tl.constexpr,
    BR: tl.constexpr,
):
    """执行一个请求窗口的 replay decode，并写入 history。

    只把本步 token 追加到长度为 L 的连续前缀，不写 canonical SSM。
    cursor 是已有前缀长度，本步从那里接着写。请求数属于运行时布局，
    不应因为 eager batch 改变而形成新的 kernel 变体。
    """
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
    n = tl.load(Cursors + req)
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)

    if PRECOMPUTE_PROJECTIONS:
        projection_dtype: tl.constexpr = (
            tl.bfloat16 if State.dtype.element_ty == tl.bfloat16 and Q.dtype.element_ty == tl.bfloat16 else tl.float32
        )
        projection_precision: tl.constexpr = "tf32" if Q.dtype.element_ty == tl.bfloat16 else "tf32x3"
        qk, projections, tt, tokens, valid, h = _project_replay_queries(
            Q,
            Kp,
            A,
            Alog,
            Bias,
            Gates,
            SQ,
            SK,
            SA,
            H,
            HV,
            K,
            L,
            BW,
            KDA,
            LOWER,
            BQK,
            end,
            hv,
            kk,
            ll,
            n,
            projection_dtype,
            projection_precision,
            slot,
            start,
            state,
        )
    else:
        projections, tt = None, None

    if PRECOMPUTE_PROJECTIONS and not KDA:
        _replay_gdn_window(
            Kp,
            Vp,
            A,
            B,
            Alog,
            Bias,
            Keys,
            Deltas,
            Gates,
            RawKeys,
            RawValues,
            Betas,
            Out,
            SK,
            SV,
            SA,
            SB,
            HV,
            KH,
            K,
            V,
            L,
            BV,
            BW,
            BR,
            end,
            h,
            hv,
            iv,
            key_slot,
            kk,
            n,
            projection_dtype,
            projection_precision,
            projections,
            qk,
            slot,
            start,
            tokens,
            tt,
            valid,
            vv,
        )
    else:
        _replay_sequential(
            Q,
            Kp,
            Vp,
            A,
            B,
            Alog,
            Bias,
            Keys,
            Deltas,
            Gates,
            RawKeys,
            RawValues,
            Betas,
            Out,
            SQ,
            SK,
            SV,
            SA,
            SB,
            H,
            HV,
            KH,
            K,
            V,
            L,
            PRECOMPUTE_PROJECTIONS,
            BW,
            KDA,
            LOWER,
            end,
            hv,
            iv,
            key_slot,
            kk,
            ll,
            n,
            projections,
            slot,
            start,
            state,
            tt,
            vv,
        )


@triton.jit
def _project_replay_queries(
    Q,
    Kp,
    A,
    Alog,
    Bias,
    Gates,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SA: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    BW: tl.constexpr,
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
    BQK: tl.constexpr,
    end,
    hv,
    kk,
    ll,
    n,
    projection_dtype: tl.constexpr,
    projection_precision: tl.constexpr,
    slot,
    start,
    state,
):
    # 预计算 verify window 的 q/k 投影。
    tt = tl.arange(0, BQK)
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
            Gates + (slot * L + ll[:, None]) * K + kk[None, :],
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
        projections = tl.dot(qk.to(projection_dtype), state.to(projection_dtype), input_precision=projection_precision)
    return qk, projections, tt, tokens, valid, h


@triton.jit
def _replay_gdn_window(
    Kp,
    Vp,
    A,
    B,
    Alog,
    Bias,
    Keys,
    Deltas,
    Gates,
    RawKeys,
    RawValues,
    Betas,
    Out,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    HV: tl.constexpr,
    KH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BV: tl.constexpr,
    BW: tl.constexpr,
    BR: tl.constexpr,
    end,
    h,
    hv,
    iv,
    key_slot,
    kk,
    n,
    projection_dtype: tl.constexpr,
    projection_precision: tl.constexpr,
    projections,
    qk,
    slot,
    start,
    tokens,
    tt,
    valid,
    vv,
):
    # GDN precompute 路径：一次投影整个窗口，再按 token 顺序写入新 history。
    # 只有 corrected-value solve 依赖 token 顺序，避免重复计算历史投影。
    jj = tl.arange(0, BR)
    fresh_tokens = start + jj - L
    old = (jj < L) & (jj < n)
    fresh = (jj >= L) & (jj < L + end - start)
    cached_keys = tl.load(
        Keys + (key_slot * L) * K + jj[:, None] * K + kk[None, :],
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
    gates = tl.load(Gates + slot * L + jj, old, 0).to(tl.float32)
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
        Deltas + (slot * L) * V + jj[:, None] * V + vv[None, :],
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
    records = n + jj - L
    in_prefix = fresh & (records < L)
    tl.store(
        Deltas + (slot * L + records[:, None]) * V + vv[None, :],
        deltas,
        in_prefix[:, None] & (vv[None, :] < V),
    )
    tl.store(
        RawValues + (slot * L + records[:, None]) * V + vv[None, :],
        values,
        in_prefix[:, None] & (vv[None, :] < V),
    )
    if iv == 0:
        if hv % (HV // KH) == 0:
            tl.store(
                Keys + (key_slot * L + records[:, None]) * K + kk[None, :],
                fresh_keys,
                in_prefix[:, None] & (kk[None, :] < K),
            )
            tl.store(
                RawKeys + (key_slot * L + records[:, None]) * K + kk[None, :],
                raw_keys,
                in_prefix[:, None] & (kk[None, :] < K),
            )
        tl.store(Gates + slot * L + records, fresh_gates, in_prefix)
        tl.store(Betas + slot * L + records, betas, in_prefix)
    return


@triton.jit
def _replay_sequential(
    Q,
    Kp,
    Vp,
    A,
    B,
    Alog,
    Bias,
    Keys,
    Deltas,
    Gates,
    RawKeys,
    RawValues,
    Betas,
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
    PRECOMPUTE_PROJECTIONS: tl.constexpr,
    BW: tl.constexpr,
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
    end,
    hv,
    iv,
    key_slot,
    kk,
    ll,
    n,
    projections,
    slot,
    start,
    state,
    tt,
    vv,
):
    # KDA、inline 以及不满足 precompute 条件的路径使用逐 token 递推。
    history = ll < n
    keys = tl.load(
        Keys + (key_slot * L) * K + ll[:, None] * K + kk[None, :],
        history[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    deltas = tl.load(
        Deltas + (slot * L) * V + ll[:, None] * V + vv[None, :],
        history[:, None] & (vv[None, :] < V),
        0,
    ).to(tl.float32)
    if KDA:
        gates = tl.load(
            Gates + (slot * L + ll[:, None]) * K + kk[None, :],
            history[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
    else:
        gates = tl.load(Gates + slot * L + ll, history, 0).to(tl.float32)
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
        record = n
        in_prefix = record < L
        tl.store(Deltas + (slot * L + record) * V + vv, d, in_prefix & (vv < V))
        tl.store(RawValues + (slot * L + record) * V + vv, v, in_prefix & (vv < V))
        if iv == 0:
            # Only the group owner writes shared K. Every CTA computes this
            # round's K locally; readers only consume earlier accepted records.
            if hv % (HV // KH) == 0:
                tl.store(Keys + (key_slot * L + record) * K + kk, k, in_prefix & (kk < K))
                tl.store(RawKeys + (key_slot * L + record) * K + kk, raw_k, in_prefix & (kk < K))
            if KDA:
                tl.store(Gates + (slot * L + record) * K + kk, g, in_prefix & (kk < K))
            else:
                tl.store(Gates + slot * L + record, g, in_prefix)
            tl.store(Betas + slot * L + record, beta, in_prefix)
        keys = tl.where((ll == n)[:, None], k[None, :], keys)
        deltas = tl.where((ll == n)[:, None], d[None, :], deltas)
        if not KDA:
            gate_prefix = tl.where(ll >= n, total_g, gate_prefix)
        history |= ll == n
        n += 1
