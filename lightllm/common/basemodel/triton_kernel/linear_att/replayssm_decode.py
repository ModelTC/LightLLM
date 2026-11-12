# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM decode kernels.

`_decode_replay` is the entry. When the accepted prefix does not fit, it folds
that prefix into the SSM and then replays the new tokens from the rounded
state. The output is either a precomputed GDN or KDA window, or a token walk.
"""

import torch
import triton
import triton.language as tl

from .replayssm_delta import _apply_accepted_prefix


def check_replay_sync_mark(
    sync_mark: torch.Tensor,
    reqs: torch.Tensor,
    value_heads: int,
    key_heads: int,
) -> None:
    """确认 SyncMark 按 [launch 行, key head] 连续存放，并且盖住这次启动的每一行。"""

    assert key_heads > 0 and value_heads % key_heads == 0, "value heads must be a multiple of key heads"
    assert sync_mark.dtype == torch.int32 and sync_mark.dim() == 2, "sync mark must be a 2D int32 tensor"
    assert sync_mark.shape[1] == key_heads, "sync mark columns must equal key heads"
    assert sync_mark.shape[0] >= reqs.numel(), "sync mark rows must cover every launch row"
    assert sync_mark.stride(1) == 1 and sync_mark.stride(0) == key_heads, "sync mark must be contiguous by key head"


@triton.jit
def _arrive_value_tiles(SyncMark, seq, hv, HV: tl.constexpr, KH: tl.constexpr, NV: tl.constexpr):
    # 同一 launch 行、同一 key head 的全部 value head 和 V tile 都读完旧 RawKeys 后，才允许写新窗口。
    # 行号不是请求槽。HOLD 会在一批里重复出现，用请求槽做下标会撞到同一格。
    # SyncMark 每行长度是 KH。最后到达的 tile 把计数写回 0，这个 0 就是放行信号。
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
def _history_coeff(rows, vec, BK: tl.constexpr):
    # rows [R, BK] · vec [BK] -> [R]. tf32x3 是 tensor core 上最接近 fp32 的输出重建。
    if BK >= 16:
        return tl.sum(tl.dot(rows, vec[:, None], input_precision="tf32x3"), 1)
    return tl.sum(rows * vec[None, :], 1)


@triton.jit
def _apply_deltas(coeff, deltas, R: tl.constexpr, BV: tl.constexpr):
    # coeff [R] · deltas [R, BV] -> [BV]. R 小于 16 时 tensor core 不能做这次点积。
    if R >= 16 and BV >= 16:
        return tl.sum(tl.dot(coeff[None, :], deltas, input_precision="tf32x3"), 0)
    return tl.sum(deltas * coeff[:, None], 0)


@triton.jit
def _decode_replay(
    Q,
    StepK,
    StepV,
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
    PRECOMPUTE_PROJECTIONS: tl.constexpr,
    BW: tl.constexpr,
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
    BQK: tl.constexpr,
    BR: tl.constexpr,
    NV: tl.constexpr,
):
    """执行一个请求窗口的 replay decode，并写入 history。

    前缀放不下本步 token 时，先在本 kernel 里把已有前缀从原始 k/v 折进 checkpoint，
    舍入成 SSM 的 dtype，再从这份状态重放本步。cursor 本 kernel 不修改，留给最后一层返回后更新。
    """
    iv, seq, hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(Reqs + seq).to(tl.int64)
    start, end = tl.load(Cu + seq), tl.load(Cu + seq + 1)
    vv = iv * BV + tl.arange(0, BV)
    if req == HOLD:
        for t in range(start, end):
            tl.store(Out + (t * HV + hv) * V + vv, 0, vv < V)
        return
    if start == end:
        # 本步没有 token。cursor 保持原值，接受 0 个 token 时长度不变。
        return
    n = tl.load(Cursors + req)

    kk = tl.arange(0, BK)
    ll = tl.arange(0, L)
    slot = req * HV + hv
    key_slot = req * KH + hv // (HV // KH)
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    # 放不下时先折回，再在同一 kernel 里重放新 token。各层只读 cursor。
    if n + (end - start) > L:
        state = _apply_accepted_prefix(state, RawKeys, RawValues, Gates, Betas, key_slot, slot, kk, vv, n, K, V, L, KDA)
        state = state.to(State.dtype.element_ty).to(tl.float32)
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
        n = 0
        # 新 token 会从槽 0 覆盖刚读过的 RawKeys。等同一 key head 的全部 value tile 都读完再往下走。
        _arrive_value_tiles(SyncMark, seq, hv, HV, KH, NV)

    if PRECOMPUTE_PROJECTIONS:
        # 输出用 tf32x3 tensor core 从冻结的 checkpoint 和 history 重建。
        # 状态不在这里更新；压缩仍用原始 k/v 做 fp32 串行折回。
        projection_dtype: tl.constexpr = tl.float32
        projection_precision: tl.constexpr = "tf32x3"
        qk, projections, tt, tokens, valid, h = _project_qk(
            Q,
            StepK,
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

    if PRECOMPUTE_PROJECTIONS and KDA:
        _kda_precomputed_window(
            StepK,
            StepV,
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
            BK,
            LOWER,
            end,
            h,
            hv,
            iv,
            key_slot,
            kk,
            n,
            projections,
            qk,
            slot,
            start,
            tt,
            vv,
        )
    elif PRECOMPUTE_PROJECTIONS:
        _gdn_precomputed_window(
            StepK,
            StepV,
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
        _replay_each_token(
            Q,
            StepK,
            StepV,
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
            KDA,
            LOWER,
            BK,
            BV,
            end,
            hv,
            iv,
            key_slot,
            kk,
            ll,
            n,
            slot,
            start,
            state,
            vv,
        )


@triton.jit
def _project_qk(
    Q,
    StepK,
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
    safe_tokens = tl.where(valid, tokens, start)
    h = hv // (HV // H)
    q_rows = tl.load(
        Q + safe_tokens[:, None] * SQ + h * K + kk[None, :],
        (valid & (tt < BW))[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    k_rows = tl.load(
        StepK + safe_tokens[:, None] * SK + h * K + kk[None, :],
        (valid & (tt >= BW))[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    # Padded query/key rows share the MMA tile. Keep them at zero so they
    # cannot change the valid rows.
    row_valid = valid[:, None]
    qk = tl.where(row_valid, q_rows + k_rows, 0)
    qk *= tl.rsqrt(tl.sum(qk * qk, axis=1) + 1.0e-6)[:, None]
    qk *= tl.where(tt < BW, K ** -0.5, 1.0)[:, None]
    qk = tl.where(row_valid, qk, 0)
    if KDA:
        history = ll < n
        gates = tl.load(
            Gates + (slot * L + ll[:, None]) * K + kk[None, :],
            history[:, None] & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        total_g = tl.sum(gates, axis=0)
        wi = tl.arange(0, BW)
        gate_ok = start + wi < end
        safe_wi = tl.where(gate_ok, wi, 0)
        gate_inputs = tl.load(
            A + (start + safe_wi[:, None]) * SA + hv * K + kk[None, :],
            gate_ok[:, None] & (kk[None, :] < K),
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
        # Padded rows stay zero: 0 * exp(garbage) is NaN and would enter the dot.
        scaled = tl.where(row_valid, qk * tl.exp(query_gates), 0)
        projections = tl.dot(
            scaled.to(projection_dtype),
            state.to(projection_dtype),
            input_precision=projection_precision,
        )
    else:
        projections = tl.dot(qk.to(projection_dtype), state.to(projection_dtype), input_precision=projection_precision)
    return qk, projections, tt, tokens, valid, h


@triton.jit
def _gdn_precomputed_window(
    StepK,
    StepV,
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
    # BR is rounded above the history length. Masked lanes must still address
    # a real slot; otherwise they read uninitialized neighbor rows into the dot.
    safe_jj = tl.where(jj < L, jj, 0)
    safe_fresh = tl.where(fresh, fresh_tokens, start)
    cached_keys = tl.load(
        Keys + (key_slot * L) * K + safe_jj[:, None] * K + kk[None, :],
        old[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    raw_keys = tl.load(
        StepK + safe_fresh[:, None] * SK + h * K + kk[None, :],
        fresh[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    fresh_keys = raw_keys * tl.rsqrt(tl.sum(raw_keys * raw_keys, axis=1) + 1.0e-6)[:, None]
    fresh_keys = tl.where(fresh[:, None], fresh_keys, 0)
    all_keys = cached_keys + fresh_keys
    values = tl.load(
        StepV + safe_fresh[:, None] * SV + hv * V + vv[None, :],
        fresh[:, None] & (vv[None, :] < V),
        0,
    ).to(tl.float32)
    log_a, bias = tl.load(Alog + hv).to(tl.float32), tl.load(Bias + hv).to(tl.float32)
    x = tl.load(A + safe_fresh * SA + hv, fresh, 0).to(tl.float32) + bias
    fresh_gates = -tl.exp(log_a) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    gates = tl.load(Gates + slot * L + safe_jj, old, 0).to(tl.float32)
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
        Deltas + (slot * L) * V + safe_jj[:, None] * V + vv[None, :],
        old[:, None] & (vv[None, :] < V),
        0,
    ).to(tl.float32)
    betas = tl.sigmoid(tl.load(B + safe_fresh * SB + hv, fresh, 0).to(tl.float32))
    for t in range(end - start):
        checkpoint_sk = tl.sum(tl.where((tt == BW + t)[:, None], projections, 0), 0)
        weights = tl.sum(tl.where((tt == BW + t)[:, None], coefficients, 0), 0)
        sk = checkpoint_sk + _apply_deltas(weights, deltas, BR, BV)
        value = tl.sum(tl.where((jj == L + t)[:, None], values, 0), 0)
        beta = tl.sum(tl.where(jj == L + t, betas, 0))
        delta = beta * (value - sk)
        deltas = tl.where((jj == L + t)[:, None], delta[None, :], deltas)
    outputs = projections + tl.dot(
        coefficients.to(projection_dtype), deltas.to(projection_dtype), input_precision=projection_precision
    )
    out_tokens = tl.where(valid, tokens, start)
    tl.store(
        Out + (out_tokens[:, None] * HV + hv) * V + vv[None, :],
        outputs,
        (valid & (tt < BW))[:, None] & (vv[None, :] < V),
    )
    records = n + jj - L
    in_prefix = fresh & (records < L)
    safe_records = tl.where(in_prefix, records, 0)
    tl.store(
        Deltas + (slot * L + safe_records[:, None]) * V + vv[None, :],
        deltas,
        in_prefix[:, None] & (vv[None, :] < V),
    )
    tl.store(
        RawValues + (slot * L + safe_records[:, None]) * V + vv[None, :],
        values,
        in_prefix[:, None] & (vv[None, :] < V),
    )
    if iv == 0:
        if hv % (HV // KH) == 0:
            tl.store(
                Keys + (key_slot * L + safe_records[:, None]) * K + kk[None, :],
                fresh_keys,
                in_prefix[:, None] & (kk[None, :] < K),
            )
            tl.store(
                RawKeys + (key_slot * L + safe_records[:, None]) * K + kk[None, :],
                raw_keys,
                in_prefix[:, None] & (kk[None, :] < K),
            )
        tl.store(Gates + slot * L + safe_records, fresh_gates, in_prefix)
        tl.store(Betas + slot * L + safe_records, betas, in_prefix)
    return


@triton.jit
def _kda_precomputed_window(
    StepK,
    StepV,
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
    BK: tl.constexpr,
    LOWER: tl.constexpr,
    end,
    h,
    hv,
    iv,
    key_slot,
    kk,
    n,
    projections,
    qk,
    slot,
    start,
    tt,
    vv,
):
    # KDA 的衰减在每个 key 通道上，不能把标量 exp 提到点积外面。
    # 每个 token 用冻结 checkpoint 的投影，加上 history 的通道衰减点积重建输出。
    # exp(G_t - G_j) 对已可见的 token 不大于 1。新 delta 仍按 token 解。
    # 状态压缩继续用原始 k/v 在 fp32 里折回。
    jj = tl.arange(0, BR)
    fresh_tokens = start + jj - L
    old = (jj < L) & (jj < n)
    fresh = (jj >= L) & (jj < L + end - start)
    safe_jj = tl.where(jj < L, jj, 0)
    safe_fresh = tl.where(fresh, fresh_tokens, start)
    cached_keys = tl.load(
        Keys + (key_slot * L) * K + safe_jj[:, None] * K + kk[None, :],
        old[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    raw_keys = tl.load(
        StepK + safe_fresh[:, None] * SK + h * K + kk[None, :],
        fresh[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    fresh_keys = raw_keys * tl.rsqrt(tl.sum(raw_keys * raw_keys, axis=1) + 1.0e-6)[:, None]
    fresh_keys = tl.where(fresh[:, None], fresh_keys, 0)
    all_keys = tl.where(old[:, None], cached_keys, 0) + fresh_keys
    values = tl.load(
        StepV + safe_fresh[:, None] * SV + hv * V + vv[None, :],
        fresh[:, None] & (vv[None, :] < V),
        0,
    ).to(tl.float32)
    hist_g = tl.load(
        Gates + (slot * L + safe_jj[:, None]) * K + kk[None, :],
        old[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    total_hist = tl.sum(hist_g, axis=0)
    hist_prefix = tl.cumsum(hist_g, axis=0)
    wi = tl.arange(0, BW)
    gate_ok = start + wi < end
    safe_wi = tl.where(gate_ok, wi, 0)
    gate_inputs = tl.load(
        A + (start + safe_wi[:, None]) * SA + hv * K + kk[None, :],
        gate_ok[:, None] & (kk[None, :] < K),
        0,
    ).to(tl.float32)
    gate_inputs += tl.load(Bias + hv * K + kk, kk < K, 0).to(tl.float32)[None, :]
    fresh_g_rows = LOWER * tl.sigmoid(tl.exp(tl.load(Alog + hv).to(tl.float32)) * gate_inputs)
    fresh_g_rows = tl.where(gate_ok[:, None], fresh_g_rows, 0)
    fresh_prefix = tl.cumsum(fresh_g_rows, axis=0)
    fresh_index = tl.broadcast_to(tl.where(fresh, jj - L, 0)[:, None], (BR, BK))
    G_pos = tl.where(old[:, None], hist_prefix, 0.0)
    G_pos = tl.where(fresh[:, None], total_hist[None, :] + tl.gather(fresh_prefix, fresh_index, 0), G_pos)
    step_g = tl.where(fresh[:, None], tl.gather(fresh_g_rows, fresh_index, 0), 0.0)
    G_query = total_hist[None, :] + tl.gather(fresh_prefix, tl.broadcast_to((tt % BW)[:, None], qk.shape), 0)
    deltas = tl.load(
        Deltas + (slot * L) * V + safe_jj[:, None] * V + vv[None, :],
        old[:, None] & (vv[None, :] < V),
        0,
    ).to(tl.float32)
    betas = tl.sigmoid(tl.load(B + safe_fresh * SB + hv, fresh, 0).to(tl.float32))
    for t in range(start, end):
        local = t - start
        q = tl.sum(tl.where((tt == local)[:, None], qk, 0), 0)
        k = tl.sum(tl.where((tt == BW + local)[:, None], qk, 0), 0)
        g_now = tl.sum(tl.where((tt == local)[:, None], G_query, 0), 0)
        causal = old | (fresh & (jj <= L + local))
        decay = tl.exp(tl.where(causal[:, None], g_now[None, :] - G_pos, 0.0))
        scaled = tl.where(causal[:, None], all_keys * decay, 0)
        hk = _history_coeff(scaled, k, BK)
        proj_k = tl.sum(tl.where((tt == BW + local)[:, None], projections, 0), 0)
        sk = proj_k + _apply_deltas(hk, deltas, BR, BV)
        value = tl.sum(tl.where((jj == L + local)[:, None], values, 0), 0)
        beta = tl.sum(tl.where(jj == L + local, betas, 0))
        delta = beta * (value - sk)
        deltas = tl.where((jj == L + local)[:, None], delta[None, :], deltas)
        hq = _history_coeff(scaled, q, BK)
        proj_q = tl.sum(tl.where((tt == local)[:, None], projections, 0), 0)
        sq = proj_q + _apply_deltas(hq, deltas, BR, BV)
        tl.store(Out + (t * HV + hv) * V + vv, sq, vv < V)
    records = n + jj - L
    in_prefix = fresh & (records < L)
    safe_records = tl.where(in_prefix, records, 0)
    tl.store(
        Deltas + (slot * L + safe_records[:, None]) * V + vv[None, :],
        deltas,
        in_prefix[:, None] & (vv[None, :] < V),
    )
    tl.store(
        RawValues + (slot * L + safe_records[:, None]) * V + vv[None, :],
        values,
        in_prefix[:, None] & (vv[None, :] < V),
    )
    if iv == 0:
        if hv % (HV // KH) == 0:
            tl.store(
                Keys + (key_slot * L + safe_records[:, None]) * K + kk[None, :],
                fresh_keys,
                in_prefix[:, None] & (kk[None, :] < K),
            )
            tl.store(
                RawKeys + (key_slot * L + safe_records[:, None]) * K + kk[None, :],
                raw_keys,
                in_prefix[:, None] & (kk[None, :] < K),
            )
        tl.store(
            Gates + (slot * L + safe_records[:, None]) * K + kk[None, :],
            step_g,
            in_prefix[:, None] & (kk[None, :] < K),
        )
        tl.store(Betas + slot * L + safe_records, betas, in_prefix)
    return


@triton.jit
def _replay_each_token(
    Q,
    StepK,
    StepV,
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
    KDA: tl.constexpr,
    LOWER: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    end,
    hv,
    iv,
    key_slot,
    kk,
    ll,
    n,
    slot,
    start,
    state,
    vv,
):
    # 逐 token 递推。precompute 走单独的窗口 kernel，不会进这里。
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
        raw_k = tl.load(StepK + t * SK + h * K + kk, kk < K, 0).to(tl.float32)
        v = tl.load(StepV + t * SV + hv * V + vv, vv < V, 0).to(tl.float32)
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
            hk = _history_coeff(keys, k, BK)
            hq = _history_coeff(keys, q, BK)
            checkpoint_sk = tl.sum(state * (k * tl.exp(total_g))[:, None], axis=0)
            checkpoint_sq = tl.sum(state * (q * tl.exp(total_g))[:, None], axis=0)
            sk = checkpoint_sk + _apply_deltas(hk, deltas, L, BV)
            sq = checkpoint_sq + _apply_deltas(hq, deltas, L, BV)
        else:
            weights = tl.where(history, tl.exp(total_g - gate_prefix), 0.0)
            hk = _history_coeff(keys, k, BK) * weights
            hq = _history_coeff(keys, q, BK) * weights
            checkpoint_sk = tl.sum(state * k[:, None], axis=0)
            checkpoint_sq = tl.sum(state * q[:, None], axis=0)
            sk = checkpoint_sk * tl.exp(total_g) + _apply_deltas(hk, deltas, L, BV)
            sq = checkpoint_sq * tl.exp(total_g) + _apply_deltas(hq, deltas, L, BV)
        d = beta * (v - sk)
        tl.store(Out + (t * HV + hv) * V + vv, sq + d * tl.sum(k * q), vv < V)
        record = n
        in_prefix = record < L
        # 被 mask 掉的 lane 仍会计算地址。cursor 已经顶到容量时必须改写 0 号槽，避免非法访问。
        safe_record = tl.where(in_prefix, record, 0)
        tl.store(Deltas + (slot * L + safe_record) * V + vv, d, in_prefix & (vv < V))
        tl.store(RawValues + (slot * L + safe_record) * V + vv, v, in_prefix & (vv < V))
        if iv == 0:
            # Only the group owner writes shared K. Every CTA computes this
            # round's K locally; readers only consume earlier accepted records.
            if hv % (HV // KH) == 0:
                tl.store(Keys + (key_slot * L + safe_record) * K + kk, k, in_prefix & (kk < K))
                tl.store(RawKeys + (key_slot * L + safe_record) * K + kk, raw_k, in_prefix & (kk < K))
            if KDA:
                tl.store(Gates + (slot * L + safe_record) * K + kk, g, in_prefix & (kk < K))
            else:
                tl.store(Gates + slot * L + safe_record, g, in_prefix)
            tl.store(Betas + slot * L + safe_record, beta, in_prefix)
        keys = tl.where((ll == n)[:, None], k[None, :], keys)
        deltas = tl.where((ll == n)[:, None], d[None, :], deltas)
        if not KDA:
            gate_prefix = tl.where(ll >= n, total_g, gate_prefix)
        history |= ll == n
        n += 1
