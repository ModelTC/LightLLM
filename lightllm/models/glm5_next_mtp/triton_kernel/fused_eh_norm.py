# SPDX-License-Identifier: Apache-2.0
# Layout adapted from vLLM's fused_eh_norm.py (5082c45853ca93ae809c58255e6126cfe20c415f).
# Copyright contributors to the vLLM project.
# Retain LightLLM's main norm, intermediate rounding and in-place updates.

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(values: tl.tensor, weight: tl.tensor, eps: float, H: tl.constexpr) -> tl.tensor:
    variance = tl.sum(values * values, axis=0) / H
    reciprocal_std = 1 / tl.sqrt(variance + eps)
    return (values * reciprocal_std) * weight


@triton.jit
def _fused_mtp_norm(
    E: tl.tensor,
    P: tl.tensor,
    WM: tl.tensor,
    WE: tl.tensor,
    WH: tl.tensor,
    O: tl.tensor,
    es0: tl.constexpr,
    es1: tl.constexpr,
    ps0: tl.constexpr,
    ps1: tl.constexpr,
    H: tl.constexpr,
    eps: float,
):
    row = tl.program_id(axis=0).to(tl.int64)
    col = tl.arange(0, H)
    e = tl.load(E + row * es0 + col * es1).to(tl.float32)
    p = tl.load(P + row * ps0 + col * ps1).to(tl.float32)
    wm = tl.load(WM + col).to(tl.float32)
    we = tl.load(WE + col).to(tl.float32)
    wh = tl.load(WH + col).to(tl.float32)
    # The second hidden norm reads the rounded output of main_norm.
    normalized_main = _norm(values=p, weight=wm, eps=eps, H=H).to(P.dtype.element_ty).to(tl.float32)
    normalized_e = _norm(values=e, weight=we, eps=eps, H=H).to(E.dtype.element_ty)
    normalized_p = _norm(values=normalized_main, weight=wh, eps=eps, H=H).to(P.dtype.element_ty)
    tl.store(E + row * es0 + col * es1, normalized_e)
    tl.store(P + row * ps0 + col * ps1, normalized_p)
    tl.store(O + row * 2 * H + col, normalized_e)
    tl.store(O + row * 2 * H + H + col, normalized_p)


def fused_mtp_norm(
    embeddings: torch.Tensor,
    previous_hidden: torch.Tensor,
    main_weight: torch.Tensor,
    embedding_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize GLM draft inputs in place and concatenate them for eh_proj."""
    assert embeddings.ndim == previous_hidden.ndim == 2
    assert embeddings.shape == previous_hidden.shape and embeddings.shape[1] == 4096
    assert embeddings.dtype == previous_hidden.dtype
    assert all(w.shape == (4096,) and w.is_contiguous() for w in (main_weight, embedding_weight, hidden_weight))
    rows, hidden = embeddings.shape
    out = torch.empty(size=(rows, 2 * hidden), device=embeddings.device, dtype=embeddings.dtype)
    if rows:
        _fused_mtp_norm[(rows,)](
            E=embeddings,
            P=previous_hidden,
            WM=main_weight,
            WE=embedding_weight,
            WH=hidden_weight,
            O=out,
            es0=embeddings.stride(0),
            es1=embeddings.stride(1),
            ps0=previous_hidden.stride(0),
            ps1=previous_hidden.stride(1),
            H=hidden,
            eps=eps,
            num_warps=8,
        )
    return out
