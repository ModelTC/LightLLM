# SPDX-License-Identifier: Apache-2.0

import torch

import triton
import triton.language as tl


@triton.jit
def _zero_position_embedding_kernel(
    embeddings,
    stride_embeddings_m,
    stride_embeddings_n,
    position_ids,
    hidden_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(position_ids + row) != 0:
        return

    offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        embeddings + row * stride_embeddings_m + offsets * stride_embeddings_n,
        0.0,
        mask=offsets < hidden_size,
    )


@torch.no_grad()
def zero_position_embedding_(embeddings: torch.Tensor, position_ids: torch.Tensor) -> None:
    """Zero MTP token embeddings at absolute position zero in place."""

    assert embeddings.is_cuda and position_ids.is_cuda
    assert embeddings.ndim == 2 and position_ids.shape == embeddings.shape[:1]
    block_n = 256
    grid = (embeddings.shape[0], triton.cdiv(embeddings.shape[1], block_n))
    _zero_position_embedding_kernel[grid](
        embeddings=embeddings,
        stride_embeddings_m=embeddings.stride(0),
        stride_embeddings_n=embeddings.stride(1),
        position_ids=position_ids,
        hidden_size=embeddings.shape[1],
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )
