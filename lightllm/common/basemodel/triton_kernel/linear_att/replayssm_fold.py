# SPDX-License-Identifier: Apache-2.0
"""Fold an accepted prefix into the canonical SSM."""

import triton
import triton.language as tl

from .replayssm_delta import _delta_rule_step


@triton.jit
def _fold_prefix(
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
    FLUSH_EXPORT: tl.constexpr = False,
    OUTPUT_STRIDES: tl.constexpr = None,
    WHEN_FULL: tl.constexpr = False,
    NEXT: tl.constexpr = 1,
):
    # 将 cursor 指向的 accepted history 应用到 canonical state。SNAPSHOT
    # 只写独立输出。WHEN_FULL 只压缩「长度 + 下一次最大宽度」放不下的请求，
    # 并在 kernel 内把这些请求的 cursor 清零，避免主机分支进不了 CUDA Graph。
    iv, row, lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(Reqs + row).to(tl.int64)
    if req == HOLD:
        return
    n = tl.load(Cursors + req)
    if WHEN_FULL:
        if n == 0 or n + NEXT <= L:
            return
    elif n == 0 and not SNAPSHOT and not FLUSH_EXPORT:
        return
    linear_layer_idx, head = lh // HV, lh % HV
    slot = (linear_layer_idx * SLOTS + req) * HV + head
    key_slot = (linear_layer_idx * SLOTS + req) * KH + head // (HV // KH)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    sp = State + slot * K * V + kk[:, None] * V + vv[None, :]
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    # cursor 可能暂时超过容量。循环只走容量内的槽，越界步不读 history，也不更新 state。
    for j in range(L):
        active = j < n
        raw_k = tl.load(RawKeys + (key_slot * L + j) * K + kk, active & (kk < K), 0).to(tl.float32)
        raw_v = tl.load(RawValues + (slot * L + j) * V + vv, active & (vv < V), 0).to(tl.float32)
        if KDA:
            g = tl.load(Gates + (slot * L + j) * K + kk, active & (kk < K), 0).to(tl.float32)
        else:
            g = tl.load(Gates + slot * L + j, active, 0).to(tl.float32)
        beta = tl.load(Betas + slot * L + j, active, 0).to(tl.float32)
        updated = _delta_rule_step(state, raw_k, raw_v, g, beta, KDA)
        state = tl.where(active, updated, state)
    mask = (kk[:, None] < K) & (vv[None, :] < V)
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
    tl.store(out, state, mask)
    if FLUSH_EXPORT:
        tl.store(sp, state, mask)
    if (FLUSH_EXPORT or WHEN_FULL) and iv == 0 and lh == 0:
        tl.store(Cursors + req, 0)
