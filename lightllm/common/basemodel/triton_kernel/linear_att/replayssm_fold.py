# SPDX-License-Identifier: Apache-2.0
"""Fold an accepted prefix into the canonical SSM."""

import triton
import triton.language as tl

from .replayssm_delta import _apply_accepted_prefix


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
    STRIDE_LAYER: tl.constexpr = 0,
    STRIDE_SLOT: tl.constexpr = 0,
    STRIDE_HEAD: tl.constexpr = 0,
    STRIDE_K: tl.constexpr = 0,
    STRIDE_V: tl.constexpr = 1,
):
    # 将 cursor 指向的 accepted history 应用到 canonical state。SNAPSHOT
    # 只写独立输出。WHEN_FULL 只压缩「长度 + 下一次最大宽度」放不下的请求。
    # cursor 清零由随后的 kernel 完成，避免和还在读 cursor 的线程块竞争。
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
    # history 是连续分配的。canonical SSM 可能带大于 int32 的层距，地址必须用真实 stride 的 int64。
    layer64 = linear_layer_idx.to(tl.int64)
    head64 = head.to(tl.int64)
    stride_layer: tl.constexpr = STRIDE_LAYER if STRIDE_LAYER != 0 else SLOTS * HV * K * V
    stride_slot: tl.constexpr = STRIDE_SLOT if STRIDE_SLOT != 0 else HV * K * V
    stride_head: tl.constexpr = STRIDE_HEAD if STRIDE_HEAD != 0 else K * V
    stride_k: tl.constexpr = STRIDE_K if STRIDE_K != 0 else V
    slot = (linear_layer_idx * SLOTS + req) * HV + head
    key_slot = (linear_layer_idx * SLOTS + req) * KH + head // (HV // KH)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    sp = (
        State
        + layer64 * stride_layer
        + req * stride_slot
        + head64 * stride_head
        + kk[:, None].to(tl.int64) * stride_k
        + vv[None, :].to(tl.int64) * STRIDE_V
    )
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    # cursor 可能暂时超过容量。循环只走容量内的槽，越界步不读 history，也不更新 state。
    state = _apply_accepted_prefix(state, RawKeys, RawValues, Gates, Betas, key_slot, slot, kk, vv, n, K, V, L, KDA)
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
    elif SNAPSHOT:
        out_slot = layer64 * HV + head64
        out = Output + out_slot * K * V + kk[:, None].to(tl.int64) * V + vv[None, :].to(tl.int64)
    else:
        out = sp
    tl.store(out, state, mask)
    if FLUSH_EXPORT:
        tl.store(sp, state, mask)
    # cursor 不在这里清。和折回并行的其他线程块还要读原来的长度，清零放到 kernel 结束之后。
