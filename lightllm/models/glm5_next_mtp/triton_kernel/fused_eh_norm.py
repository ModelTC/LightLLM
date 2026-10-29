# SPDX-License-Identifier: Apache-2.0
# MTP norm/cat fusion follows the layout idea in vLLM's fused_eh_norm.py:
# https://github.com/vllm-project/vllm/blob/5082c45853ca93ae809c58255e6126cfe20c415f/vllm/models/glm5next/nvidia/ops/fused_eh_norm.py
# Copyright contributors to the vLLM project.
# This variant retains LightLLM's additional main norm, all in-place writes,
# and the intermediate dtype rounding of the pre-optimization implementation.

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(values, weight, eps, H: tl.constexpr):
    variance = tl.sum(values * values, axis=0) / H
    reciprocal_std = 1 / tl.sqrt(variance + eps)
    return (values * reciprocal_std) * weight


@triton.jit
def _kernel(
    E,
    P,
    WM,
    WE,
    WH,
    O,
    es0: tl.constexpr,
    es1: tl.constexpr,
    ps0: tl.constexpr,
    ps1: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < H
    e = tl.load(E + row * es0 + col * es1, mask, 0).to(tl.float32)
    p = tl.load(P + row * ps0 + col * ps1, mask, 0).to(tl.float32)
    wm = tl.load(WM + col, mask, 0).to(tl.float32)
    we = tl.load(WE + col, mask, 0).to(tl.float32)
    wh = tl.load(WH + col, mask, 0).to(tl.float32)
    # main_norm previously stored into P before hnorm reloaded that BF16 tensor.
    normalized_main = _norm(p, wm, eps, H).to(P.dtype.element_ty).to(tl.float32)
    normalized_e = _norm(e, we, eps, H).to(E.dtype.element_ty)
    normalized_p = _norm(normalized_main, wh, eps, H).to(P.dtype.element_ty)
    tl.store(E + row * es0 + col * es1, normalized_e, mask)
    tl.store(P + row * ps0 + col * ps1, normalized_p, mask)
    tl.store(O + row * 2 * H + col, normalized_e, mask)
    tl.store(O + row * 2 * H + H + col, normalized_p, mask)


def fused_mtp_norm(embeddings, previous_hidden, main_weight, embedding_weight, hidden_weight, eps):
    assert embeddings.ndim == previous_hidden.ndim == 2
    assert embeddings.shape == previous_hidden.shape
    assert embeddings.dtype == previous_hidden.dtype
    n, h = embeddings.shape
    assert all(w.shape == (h,) and w.is_contiguous() for w in [main_weight, embedding_weight, hidden_weight])
    out = torch.empty((n, 2 * h), device=embeddings.device, dtype=embeddings.dtype)
    if n:
        _kernel[(n,)](
            embeddings,
            previous_hidden,
            main_weight,
            embedding_weight,
            hidden_weight,
            out,
            *embeddings.stride(),
            *previous_hidden.stride(),
            h,
            triton.next_power_of_2(h),
            eps,
            num_warps=8,
        )
    return out
