# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LightLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, The vLLM team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES.
# Triton implementation matching the router semantics adapted from vLLM commit
# 36d2a5086bae12d0c0b311607373e4fc2d1036aa.

import torch
import triton
import triton.language as tl

from .launch import pdl_kwargs


@triton.jit
def _sqrt_softplus(x):
    score = tl.sqrt(tl.maximum(x, 0) + tl.log(1 + tl.exp(-tl.abs(x))))
    return tl.where(score == score, score, 0)


@triton.jit
def _hash_route(logits, input_ids, tid2eid, rows, valid, slots, NUM_EXPERTS: tl.constexpr):
    tokens = tl.load(input_ids + rows, mask=valid, other=0)
    ids = tl.load(tid2eid + tokens[:, None] * 6 + slots[None, :], mask=valid[:, None] & (slots[None, :] < 6), other=0)
    logit_values = tl.load(
        logits + rows[:, None] * NUM_EXPERTS + ids, mask=valid[:, None] & (slots[None, :] < 6), other=0
    )
    return ids, tl.where(slots[None, :] < 6, _sqrt_softplus(logit_values), 0)


@triton.jit
def _router_kernel(
    logits,
    topk_weights,
    topk_indices,
    correction_bias,
    input_ids,
    tid2eid,
    bias_vl,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    IMAGE_TOKEN_START: tl.constexpr,
    HAS_HASH_ROUTING: tl.constexpr,
    HAS_VISION_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = rows < num_tokens
    slots = tl.arange(0, 8)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_HASH_ROUTING and not HAS_VISION_BIAS:
        ids, selected = _hash_route(logits, input_ids, tid2eid, rows, valid, slots, NUM_EXPERTS)
    else:
        image = tl.full((BLOCK_R,), False, tl.int1)
        if HAS_VISION_BIAS:
            tokens = tl.load(input_ids + rows, mask=valid, other=0)
            image = tokens >= IMAGE_TOKEN_START
        dense = valid
        if HAS_HASH_ROUTING:
            dense = valid & image
        experts = tl.arange(0, BLOCK_E)
        logit_values = tl.load(
            logits + rows[:, None] * NUM_EXPERTS + experts[None, :],
            mask=dense[:, None] & (experts[None, :] < NUM_EXPERTS),
            other=0,
        )
        scores = _sqrt_softplus(logit_values)
        bias = tl.load(correction_bias + experts, mask=experts < NUM_EXPERTS, other=0)
        selection = scores + bias[None, :]
        if HAS_VISION_BIAS:
            image_bias = tl.load(bias_vl + experts, mask=experts < NUM_EXPERTS, other=0)
            selection = tl.where(image[:, None], scores + image_bias[None, :], selection)
        remaining = tl.where(dense[:, None] & (experts[None, :] < NUM_EXPERTS), selection, -float("inf"))
        selected = tl.full((BLOCK_R, 8), 0, tl.float32)
        ids = tl.full((BLOCK_R, 8), 0, tl.int64)
        for slot in tl.static_range(6):
            _, expert = tl.max(remaining, axis=1, return_indices=True)
            value = tl.sum(tl.where(experts[None, :] == expert[:, None], scores, 0), axis=1)
            selected = tl.where(slots[None, :] == slot, value[:, None], selected)
            ids = tl.where(slots[None, :] == slot, expert[:, None].to(tl.int64), ids)
            remaining = tl.where(experts[None, :] == expert[:, None], -float("inf"), remaining)
        if HAS_HASH_ROUTING:
            hash_ids, hash_weights = _hash_route(logits, input_ids, tid2eid, rows, valid & ~image, slots, NUM_EXPERTS)
            ids = tl.where(image[:, None], ids, hash_ids)
            selected = tl.where(image[:, None], selected, hash_weights)
    total = tl.sum(selected, axis=1)
    selected = selected * ROUTED_SCALING_FACTOR / tl.where(total[:, None] > 0, total[:, None], 1)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    mask = valid[:, None] & (slots[None, :] < 6)
    tl.store(topk_weights + rows[:, None] * 6 + slots[None, :], selected, mask=mask)
    tl.store(topk_indices + rows[:, None] * 6 + slots[None, :], ids, mask=mask)


@torch.no_grad()
def topk_softplus_sqrt(
    topk_weights,
    topk_indices,
    logits,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
    bias_vl=None,
    image_token_start=0,
):
    """Route FP32 logits [T, experts] to six experts, writing weights and int64 IDs.

    Hash layers use tid2eid [vocab, 6]. Vision tokens select by bias_vl; other
    dense rows use correction_bias. Selected scores normalize to routed_scaling_factor.
    """
    rows = logits.shape[0]
    if rows == 0:
        return
    hash_only = tid2eid is not None and bias_vl is None
    block_r, warps = (16, 8) if hash_only and rows >= 256 else (1, 1)
    _router_kernel[(triton.cdiv(rows, block_r),)](
        logits,
        topk_weights,
        topk_indices,
        correction_bias if correction_bias is not None else logits,
        input_ids,
        tid2eid,
        bias_vl,
        rows,
        NUM_EXPERTS=logits.shape[1],
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        IMAGE_TOKEN_START=image_token_start,
        HAS_HASH_ROUTING=tid2eid is not None,
        HAS_VISION_BIAS=bias_vl is not None,
        BLOCK_E=triton.next_power_of_2(logits.shape[1]),
        BLOCK_R=block_r,
        num_warps=warps,
        num_stages=1,
        **pdl_kwargs(logits.device),
    )
