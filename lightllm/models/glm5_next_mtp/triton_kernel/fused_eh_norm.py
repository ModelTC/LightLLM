# SPDX-License-Identifier: Apache-2.0
# Layout adapted from vLLM's fused_eh_norm.py (5082c45853ca93ae809c58255e6126cfe20c415f).
# Copyright contributors to the vLLM project.
# Retain LightLLM's main norm, intermediate rounding and in-place updates.

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(values: tl.tensor, weight: tl.tensor, eps: float, hidden_size: tl.constexpr) -> tl.tensor:
    variance = tl.sum(values * values, axis=0) / hidden_size
    reciprocal_std = 1 / tl.sqrt(variance + eps)
    return (values * reciprocal_std) * weight


@triton.jit
def _fused_mtp_norm_kernel(
    embeddings: tl.tensor,
    previous_hidden: tl.tensor,
    main_norm_weight: tl.tensor,
    embedding_norm_weight: tl.tensor,
    hidden_norm_weight: tl.tensor,
    output: tl.tensor,
    embeddings_stride_0: tl.constexpr,
    embeddings_stride_1: tl.constexpr,
    previous_hidden_stride_0: tl.constexpr,
    previous_hidden_stride_1: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: float,
):
    token_index = tl.program_id(axis=0).to(tl.int64)
    hidden_index = tl.arange(0, hidden_size)
    embedding_values = tl.load(embeddings + token_index * embeddings_stride_0 + hidden_index * embeddings_stride_1).to(
        tl.float32
    )
    previous_hidden_values = tl.load(
        previous_hidden + token_index * previous_hidden_stride_0 + hidden_index * previous_hidden_stride_1
    ).to(tl.float32)
    main_norm_scale = tl.load(main_norm_weight + hidden_index).to(tl.float32)
    embedding_norm_scale = tl.load(embedding_norm_weight + hidden_index).to(tl.float32)
    hidden_norm_scale = tl.load(hidden_norm_weight + hidden_index).to(tl.float32)
    # The second hidden norm reads the rounded output of main_norm.
    main_normalized_hidden = (
        _rms_norm(values=previous_hidden_values, weight=main_norm_scale, eps=eps, hidden_size=hidden_size)
        .to(previous_hidden.dtype.element_ty)
        .to(tl.float32)
    )
    normalized_embeddings = _rms_norm(
        values=embedding_values, weight=embedding_norm_scale, eps=eps, hidden_size=hidden_size
    ).to(embeddings.dtype.element_ty)
    normalized_previous_hidden = _rms_norm(
        values=main_normalized_hidden, weight=hidden_norm_scale, eps=eps, hidden_size=hidden_size
    ).to(previous_hidden.dtype.element_ty)
    tl.store(
        embeddings + token_index * embeddings_stride_0 + hidden_index * embeddings_stride_1,
        normalized_embeddings,
    )
    tl.store(
        previous_hidden + token_index * previous_hidden_stride_0 + hidden_index * previous_hidden_stride_1,
        normalized_previous_hidden,
    )
    tl.store(output + token_index * 2 * hidden_size + hidden_index, normalized_embeddings)
    tl.store(
        output + token_index * 2 * hidden_size + hidden_size + hidden_index,
        normalized_previous_hidden,
    )


def fused_mtp_norm(
    embeddings: torch.Tensor,
    previous_hidden: torch.Tensor,
    main_weight: torch.Tensor,
    embedding_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply the three GLM MTP RMS norms in place and concatenate the results."""
    num_tokens, hidden_size = embeddings.shape
    output = torch.empty(size=(num_tokens, 2 * hidden_size), device=embeddings.device, dtype=embeddings.dtype)
    _fused_mtp_norm_kernel[(num_tokens,)](
        embeddings=embeddings,
        previous_hidden=previous_hidden,
        main_norm_weight=main_weight,
        embedding_norm_weight=embedding_weight,
        hidden_norm_weight=hidden_weight,
        output=output,
        embeddings_stride_0=embeddings.stride(0),
        embeddings_stride_1=embeddings.stride(1),
        previous_hidden_stride_0=previous_hidden.stride(0),
        previous_hidden_stride_1=previous_hidden.stride(1),
        hidden_size=hidden_size,
        eps=eps,
        num_warps=8,
    )
    return output
