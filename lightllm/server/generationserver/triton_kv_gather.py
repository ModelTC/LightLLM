"""Fused CUDA gather for dense generation-side Prefix KV snapshots."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_dense_prefix_kv_kernel(
    source,
    gather_indexes,
    output,
    source_stride_layer,
    source_stride_token,
    source_stride_head,
    source_stride_dim,
    output_stride_layer,
    output_stride_kv,
    output_stride_token,
    output_stride_head,
    output_stride_dim,
    token_count,
    head_count,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    layer = tl.program_id(0).to(tl.int64)
    kv_kind = tl.program_id(1).to(tl.int64)
    block = tl.program_id(2).to(tl.int64)

    head_width = head_count * head_dim
    flat = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = flat < token_count * head_width
    packed_token = flat // head_width
    head_element = flat % head_width
    head = head_element // head_dim
    dim = head_element % head_dim
    source_token = tl.load(gather_indexes + packed_token, mask=mask, other=0).to(
        tl.int64
    )

    source_offsets = (
        layer * source_stride_layer
        + source_token * source_stride_token
        + (kv_kind * head_count + head) * source_stride_head
        + dim * source_stride_dim
    )
    output_offsets = (
        layer * output_stride_layer
        + kv_kind * output_stride_kv
        + packed_token * output_stride_token
        + head * output_stride_head
        + dim * output_stride_dim
    )
    value = tl.load(source + source_offsets, mask=mask)
    tl.store(output + output_offsets, value, mask=mask)


def triton_gather_dense_prefix_kv(
    source: torch.Tensor,
    gather_indexes: torch.Tensor,
    *,
    head_count: int,
) -> torch.Tensor:
    """Gather all layers and both K/V planes with one Triton launch.

    The target layout is ``[layer, K-or-V, token, head, dim]``. Selecting one
    layer and one K/V plane therefore yields the contiguous ``[token, head,
    dim]`` tensors expected by the consumer attention implementations.
    """

    if not source.is_cuda or not gather_indexes.is_cuda:
        raise ValueError("Triton Prefix KV gather requires CUDA tensors")
    if source.ndim != 4:
        raise ValueError("Prefix KV pool must have four dimensions")
    if gather_indexes.ndim != 1 or gather_indexes.dtype != torch.int64:
        raise TypeError("Prefix KV gather indexes must be a 1-D int64 tensor")
    if head_count <= 0 or source.shape[2] != 2 * head_count:
        raise ValueError("Prefix KV pool does not match its K/V head count")

    layer_count, _, _, head_dim = source.shape
    token_count = gather_indexes.numel()
    output = torch.empty(
        (layer_count, 2, token_count, head_count, head_dim),
        dtype=source.dtype,
        device=source.device,
    )
    block_size = 256
    grid = (
        layer_count,
        2,
        triton.cdiv(token_count * head_count * head_dim, block_size),
    )
    _gather_dense_prefix_kv_kernel[grid](
        source,
        gather_indexes,
        output,
        *source.stride(),
        *output.stride(),
        token_count,
        head_count,
        head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


__all__ = ["triton_gather_dense_prefix_kv"]
