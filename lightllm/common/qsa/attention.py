from typing import Optional

import torch
import triton

from .attention_kernels import _attention, _reduce


@torch.no_grad()
def indexed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    scale: Optional[float] = None,
    splits: Optional[int] = None,
    selection_lengths: Optional[torch.Tensor] = None,
    short_rows_direct: bool = False,
    skip_empty_splits: bool = False,
) -> torch.Tensor:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3 or indices.ndim != 2:
        raise ValueError("Indexed attention requires rank-three Q/K/V and rank-two indices")
    rows, q_heads, dim = query.shape
    if not q_heads or not dim or not key.shape[1] or q_heads % key.shape[1]:
        raise ValueError("Indexed attention query heads must be divisible by KV heads")
    if key.shape[0:2] != value.shape[0:2] or key.shape[2] != dim or not value.shape[2]:
        raise ValueError("Indexed attention key/value dimensions do not match")
    for tensor in (query, key, value):
        if not tensor.is_cuda or tensor.dtype != query.dtype or tensor.device != query.device:
            raise ValueError("Indexed attention Q/K/V must be CUDA tensors with matching dtype/device")
    if not query.is_contiguous():
        raise ValueError("Indexed attention query must be contiguous")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Indexed attention requires floating point inputs")
    if (
        indices.shape[0] != rows
        or indices.dtype != torch.int32
        or indices.device != query.device
        or not indices.is_contiguous()
    ):
        raise ValueError("Indexed attention indices must be a contiguous int32 matrix on the query device")
    selected = indices.shape[1]
    if selection_lengths is not None and (
        selection_lengths.shape != (rows,)
        or selection_lengths.dtype != torch.int32
        or selection_lengths.device != query.device
        or not selection_lengths.is_contiguous()
    ):
        raise ValueError("Selection lengths must be a contiguous int32 vector on the query device")
    kv_heads, value_dim = value.shape[1:]
    output = torch.empty((rows, q_heads, value_dim), dtype=query.dtype, device=query.device)
    if not rows:
        return output
    if not selected:
        return output.zero_()
    if splits is None:
        splits = min(32, max(1, triton.cdiv(64, rows * kv_heads)), triton.cdiv(selected, 64))
    if splits < 1:
        raise ValueError("Indexed attention split count must be positive")
    partial = (
        torch.empty((rows, q_heads, splits, value_dim), dtype=torch.float32, device=query.device)
        if splits > 1
        else None
    )
    lse = torch.empty((rows, q_heads, splits), dtype=torch.float32, device=query.device) if splits > 1 else None
    _attention[(rows, kv_heads, splits)](
        query,
        key,
        value,
        indices,
        selection_lengths,
        partial,
        lse,
        output,
        q_heads,
        kv_heads,
        *key.stride(),
        *value.stride(),
        q_heads // kv_heads,
        dim,
        value_dim,
        selected,
        splits,
        dim**-0.5 if scale is None else scale,
        max(16, triton.next_power_of_2(q_heads // kv_heads)),
        max(16, triton.next_power_of_2(dim)),
        max(16, triton.next_power_of_2(value_dim)),
        64,
        short_rows_direct,
        skip_empty_splits,
    )
    if splits > 1:
        _reduce[(rows, q_heads)](
            selection_lengths,
            partial,
            lse,
            output,
            q_heads,
            value_dim,
            splits,
            triton.next_power_of_2(splits),
            triton.next_power_of_2(value_dim),
            selected,
            64,
            short_rows_direct,
            skip_empty_splits,
        )
    return output
