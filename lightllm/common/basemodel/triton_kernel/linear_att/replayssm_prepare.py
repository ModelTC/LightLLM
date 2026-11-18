# SPDX-License-Identifier: Apache-2.0
"""Prepare and fold the double-buffered ReplaySSM history.

The decode kernel only reads the selected history half.  When the accepted
prefix would cross the end of that half, this module toggles the cursor to the
other half and folds the old half into the canonical SSM before decode starts.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    """Toggle the history half and collect rows that need a separate fold."""

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
    next_cursor = (cursor & (2 * L)) ^ (2 * L)
    tl.store(Cursors + req, next_cursor, valid & overflow)
    needs_fold = valid & (length > 0) & overflow
    offset = tl.cumsum(needs_fold.to(tl.int32), axis=0) - 1
    tl.store(Active + offset, seq, needs_fold)
    count = tl.sum(needs_fold.to(tl.int32), axis=0)

    # The main branch uses a separate fold for every overflow row.  This is
    # the safe double-buffer path and removes the decode-time global barrier.
    tl.store(ForwardCursors + req, tl.where(overflow, next_cursor, cursor), valid)
    tl.store(Active + N, count)


@triton.jit
def _replay_update(state, raw_k, raw_v, g, beta, KDA: tl.constexpr):
    raw_k /= tl.sqrt(tl.sum(raw_k * raw_k) + 1.0e-6)
    decay = tl.exp(g)
    if KDA:
        state *= decay[:, None]
        delta = beta * (raw_v - tl.sum(state * raw_k[:, None], 0))
        state += raw_k[:, None] * delta[None, :]
    else:
        sk = tl.sum(state * raw_k[:, None], 0)
        delta = beta * (raw_v - decay * sk)
        state = state * decay + raw_k[:, None] * delta[None, :]
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
    """Fold all rows collected by ``_prepare_fold`` into the SSM state."""

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
            raw_k = tl.load(
                RawKeys + linear_layer_idx * RS + (key_slot * (2 * L) + base + j) * K + kk,
                kk < K,
                0,
            ).to(tl.float32)
            raw_v = tl.load(
                RawValues + linear_layer_idx * VS + (slot * (2 * L) + base + j) * V + vv,
                vv < V,
                0,
            ).to(tl.float32)
            if KDA:
                gate = tl.load(
                    Gates + linear_layer_idx * GS + (slot * (2 * L) + base + j) * K + kk,
                    kk < K,
                    0,
                ).to(tl.float32)
            else:
                gate = tl.load(
                    Gates + linear_layer_idx * GS + slot * (2 * L) + base + j
                ).to(tl.float32)
            beta = tl.load(
                Betas + linear_layer_idx * BS + slot * (2 * L) + base + j
            ).to(tl.float32)
            state = _replay_update(state, raw_k, raw_v, gate, beta, KDA)
        # Match the canonical checkpoint precision before the next replay.
        state = state.to(State.dtype.element_ty).to(tl.float32)
        tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
        work += tl.num_programs(0)


def _fold_program_count(state, req_count: int, value_dim: int, value_heads: int) -> int:
    if state.ssm_state.is_cuda:
        fold_programs = getattr(state, "fold_programs", None)
        if fold_programs is None:
            fold_programs = 8 * torch.cuda.get_device_properties(state.ssm_state.device).multi_processor_count
        return max(1, min(fold_programs, triton.cdiv(value_dim, 32) * req_count * state.ssm_state.shape[0] * value_heads))
    return 0


def prepare_decode(state, reqs: torch.Tensor, cu_seqlens: torch.Tensor | None) -> None:
    """Prepare one decode window and fold overflow rows before layer replay."""

    if reqs.numel() == 0:
        return
    assert reqs.is_cuda, "double-buffer replay preparation requires CUDA inputs"
    if cu_seqlens is None:
        cu_seqlens = torch.arange(reqs.numel() + 1, dtype=torch.int32, device=reqs.device)
    else:
        assert cu_seqlens.is_cuda, "double-buffer replay preparation requires CUDA inputs"
        assert cu_seqlens.numel() == reqs.numel() + 1
    positions = torch.empty_like(reqs)
    active = torch.empty(reqs.numel() + 1, dtype=torch.int32, device=reqs.device)
    block = triton.next_power_of_2(max(1, reqs.numel()))
    _prepare_fold[(1,)](
        reqs,
        state.cursors,
        positions,
        state.forward_cursors,
        cu_seqlens,
        active,
        reqs.numel(),
        state.hold_req_id,
        state.capacity,
        True,
        block,
        state.verify_width,
        num_warps=4,
    )
    layers, _, hv, kd, vd = state.ssm_state.shape
    programs = _fold_program_count(state, reqs.numel(), vd, hv)
    if programs == 0:
        return
    _fold_history[(programs,)](
        state.ssm_state,
        state.raw_keys,
        state.raw_values,
        state.gates,
        state.betas,
        reqs,
        positions,
        active,
        reqs.numel(),
        layers,
        hv,
        state.num_key_heads,
        kd,
        vd,
        state.capacity,
        triton.next_power_of_2(kd),
        32,
        state.ssm_state.stride(0),
        state.raw_keys.stride(0),
        state.raw_values.stride(0),
        state.gates.stride(0),
        state.betas.stride(0),
        state.kda,
        num_warps=2,
        num_stages=1,
    )


def fold_current_history(state, reqs: torch.Tensor) -> None:
    """Materialize the currently accepted history for page/export operations."""

    if reqs.numel() == 0:
        return
    assert reqs.is_cuda, "double-buffer replay folding requires CUDA inputs"
    positions = state.cursors.index_select(0, reqs.long()).contiguous()
    active = torch.empty(reqs.numel() + 1, dtype=torch.int32, device=reqs.device)
    active[:-1] = torch.arange(reqs.numel(), dtype=torch.int32, device=reqs.device)
    active[-1] = reqs.numel()
    layers, _, hv, kd, vd = state.ssm_state.shape
    programs = _fold_program_count(state, reqs.numel(), vd, hv)
    if programs == 0:
        return
    _fold_history[(programs,)](
        state.ssm_state,
        state.raw_keys,
        state.raw_values,
        state.gates,
        state.betas,
        reqs,
        positions,
        active,
        reqs.numel(),
        layers,
        hv,
        state.num_key_heads,
        kd,
        vd,
        state.capacity,
        triton.next_power_of_2(kd),
        32,
        state.ssm_state.stride(0),
        state.raw_keys.stride(0),
        state.raw_values.stride(0),
        state.gates.stride(0),
        state.betas.stride(0),
        state.kda,
        num_warps=2,
        num_stages=1,
    )
