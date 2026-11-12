# SPDX-License-Identifier: Apache-2.0
"""Fold an accepted prefix into the canonical SSM."""

import triton
import triton.language as tl

from .replayssm_delta import _apply_accepted_prefix


@triton.jit
def _fold_prefix(
    State,
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
    KDA: tl.constexpr,
    STRIDE_LAYER: tl.constexpr,
    STRIDE_SLOT: tl.constexpr,
    STRIDE_HEAD: tl.constexpr,
    STRIDE_K: tl.constexpr,
    STRIDE_V: tl.constexpr,
):
    # 将 cursor 指向的 accepted history 写回 canonical state。
    iv, row, lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(Reqs + row).to(tl.int64)
    if req == HOLD:
        return
    prefix_len = tl.load(Cursors + req)
    if prefix_len == 0:
        return
    linear_layer_idx, head = lh // HV, lh % HV
    # history 是连续分配的。canonical SSM 可能带大于 int32 的层距，地址必须用真实 stride 的 int64。
    layer64 = linear_layer_idx.to(tl.int64)
    head64 = head.to(tl.int64)
    slot = (linear_layer_idx * SLOTS + req) * HV + head
    key_slot = (linear_layer_idx * SLOTS + req) * KH + head // (HV // KH)
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    sp = (
        State
        + layer64 * STRIDE_LAYER
        + req * STRIDE_SLOT
        + head64 * STRIDE_HEAD
        + kk[:, None].to(tl.int64) * STRIDE_K
        + vv[None, :].to(tl.int64) * STRIDE_V
    )
    state = tl.load(sp, (kk[:, None] < K) & (vv[None, :] < V), 0).to(tl.float32)
    # cursor 可能暂时超过容量。循环只走容量内的槽，越界步不读 history，也不更新 state。
    state = _apply_accepted_prefix(
        state, RawKeys, RawValues, Gates, Betas, key_slot, slot, kk, vv, prefix_len, K, V, L, KDA
    )
    tl.store(sp, state, (kk[:, None] < K) & (vv[None, :] < V))
    # cursor 不在这里清。和折回并行的其他线程块还要读原来的长度，清零放到 kernel 结束之后。


def fold_prefix(state, reqs, run_config=None) -> None:
    """把 cursor 覆盖的已接受前缀折进 canonical SSM。

    没传入 run_config 时用 BV=32、4 个 warp、3 个 stage。
    """

    if run_config is None:
        run_config = {"BV": 32, "num_warps": 4, "num_stages": 3}
    bv = run_config["BV"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]
    ssm = state.ssm_state
    layers, slots, hv, k, v = ssm.shape
    layer_stride, slot_stride, head_stride, key_stride, value_stride = ssm.stride()
    _fold_prefix[(triton.cdiv(v, bv), reqs.numel(), layers * hv)](
        ssm,
        state.raw_keys,
        state.raw_values,
        state.gates,
        state.betas,
        state.cursors,
        reqs,
        slots,
        hv,
        state.num_key_heads,
        k,
        v,
        state.capacity,
        state.hold_req_id,
        triton.next_power_of_2(k),
        bv,
        state.kda,
        STRIDE_LAYER=layer_stride,
        STRIDE_SLOT=slot_stride,
        STRIDE_HEAD=head_stride,
        STRIDE_K=key_stride,
        STRIDE_V=value_stride,
        num_warps=num_warps,
        num_stages=num_stages,
    )
