# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM history preparation and accepted-prefix folding kernels."""

import triton
import triton.language as tl

from .replayssm_common import _replay_update


@triton.jit(do_not_specialize=["N"])
def _accept_updates(
    cursors,
    reqs,
    accepted_state_indices,
    N,
    HOLD: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # N 是运行时请求数；BLOCK/WIDTH 仍是编译期参数，避免 eager batch 变化触发重新编译。
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(reqs + i, i < N, HOLD)
    valid = (i < N) & (r != HOLD)
    cursor = tl.load(cursors + r, valid, 0)
    count = tl.load(accepted_state_indices + r, valid, 0) + 1 if WIDTH > 1 else 1
    tl.store(cursors + r, cursor + count, valid)


@triton.jit
def _use_separate_fold(valid, length, count, N, L: tl.constexpr, WIDTH: tl.constexpr):
    # 只决定本轮是否单独启动 fold；真正的 history 合并由 _fold_history 完成。
    small_batch: tl.constexpr = 12
    dense_batch: tl.constexpr = 256
    live = tl.sum((valid & (length > 0)).to(tl.int32), axis=0)
    split = (count > 0) & ((live <= small_batch) | (2 * count <= live))
    if L == 8 and WIDTH == 4:
        if N >= dense_batch:
            short = tl.sum((valid & (length > 0) & (length < WIDTH)).to(tl.int32), axis=0)
            split = split & ((live < dense_batch) | (short > 0))
    return split


@triton.jit(do_not_specialize=["N"])
def _prepare_fold(
    Reqs,
    Cursors,
    Positions,
    ForwardCursors,
    Cu,
    Active,
    N,
    HOLD: tl.constexpr,
    L: tl.constexpr,
    VARLEN: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
):
    # 输出 Positions/ForwardCursors 供 decode 使用，Active/N 供后续 fold
    # 使用；只改写本轮请求对应的 cursor，不触碰 canonical state。
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
    split = _use_separate_fold(valid, length, count, N, L, WIDTH)
    tl.store(ForwardCursors + req, tl.where(overflow & split, (cursor & (2 * L)) ^ (2 * L), cursor), valid)
    tl.store(Active + N, tl.where(split, count, 0))


@triton.jit(do_not_specialize=["N"])
def _fold_history(
    State,
    RawKeys,
    RawValues,
    Gates,
    Betas,
    Reqs,
    Cursors,
    Active,
    N,
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
    # 只消费 Active 中的 fold 请求，把 history 合并回 canonical state；
    # cursor 的 phase 已由 _prepare_fold 切换，写入后新 history 从同一半区继续。
    count = tl.load(Active + N)
    work = tl.program_id(0)
    tiles: tl.constexpr = (V + BV - 1) // BV
    total = count * LAYERS * HV * tiles
    while work < total:
        iv = work % tiles
        slot_work = work // tiles
        seq = tl.load(Active + slot_work % count)
        layer_hv = slot_work // count
        linear_layer_idx, hv = (layer_hv // HV).to(tl.int64), layer_hv % HV
        req = tl.load(Reqs + seq).to(tl.int64)
        cursor = tl.load(Cursors + seq)
        n = cursor % (2 * L)
        base = (cursor // (2 * L)) * L
        kk = tl.arange(0, BK)
        vv = iv * BV + tl.arange(0, BV)
        slot = req * HV + hv
        key_slot = req * KH + hv // (HV // KH)
        sp = State + linear_layer_idx * SS + slot * K * V + kk[:, None] * V + vv[None, :]
        state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
        for j in range(n):
            raw_k = tl.load(RawKeys + linear_layer_idx * RS + (key_slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(
                tl.float32
            )
            raw_v = tl.load(RawValues + linear_layer_idx * VS + (slot * (2 * L) + base + j) * V + vv, vv < V, 0).to(
                tl.float32
            )
            if KDA:
                g = tl.load(Gates + linear_layer_idx * GS + (slot * (2 * L) + base + j) * K + kk, kk < K, 0).to(
                    tl.float32
                )
            else:
                g = tl.load(Gates + linear_layer_idx * GS + slot * (2 * L) + base + j).to(tl.float32)
            beta = tl.load(Betas + linear_layer_idx * BS + slot * (2 * L) + base + j).to(tl.float32)
            state = _replay_update(state, raw_k, raw_v, g, beta, KDA)
        # Start the new history from the same rounded checkpoint future calls load.
        state = state.to(State.dtype.element_ty).to(tl.float32)
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
        work += tl.num_programs(0)
