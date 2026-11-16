# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kda_state(
    pool,
    packed,
    indices,
    cu_seqlens,
    pool_slots: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    stride_slot: tl.constexpr,
    stride_head: tl.constexpr,
    stride_key: tl.constexpr,
    stride_value: tl.constexpr,
    STORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(indices + seq)
    slot = tl.where(slot < 0, slot + pool_slots, slot)
    active = True
    if STORE:
        active = tl.load(cu_seqlens + seq + 1) > tl.load(cu_seqlens + seq)
    if active:
        key = (tile // tl.cdiv(dim, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
        value = (tile % tl.cdiv(dim, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
        mask = (key[:, None] < dim) & (value[None, :] < dim)
        canonical = (
            pool + slot * stride_slot + head * stride_head + key[:, None] * stride_key + value[None, :] * stride_value
        )
        transposed = packed + ((seq * heads + head) * dim + value[None, :]) * dim + key[:, None]
        if STORE:
            tl.store(canonical, tl.load(transposed, mask, 0), mask)
        else:
            tl.store(transposed, tl.load(canonical, mask, 0), mask)


def gather_kda_state(states, indices):
    _, heads, key_dim, value_dim = states.shape
    assert key_dim == value_dim == 128
    packed = torch.empty((indices.numel(), heads, value_dim, key_dim), device=states.device, dtype=states.dtype)
    if indices.numel():
        _copy_kda_state[(indices.numel(), heads, 16)](
            states, packed, indices, None, states.shape[0], heads, key_dim, *states.stride(), STORE=False, BLOCK=32
        )
    return packed


def scatter_kda_state(packed, states, indices, cu_seqlens):
    if indices.numel():
        _copy_kda_state[(indices.numel(), states.shape[1], 16)](
            states,
            packed,
            indices,
            cu_seqlens,
            states.shape[0],
            states.shape[1],
            states.shape[2],
            *states.stride(),
            STORE=True,
            BLOCK=32,
        )
