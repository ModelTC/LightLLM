"""Explicit FP32 attention over a dense snapshot's local layout."""

from __future__ import annotations

import torch


def fp32_paged_prefix_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    max_q_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.dtype != torch.float32:
        raise TypeError("FP32 paged-prefix attention requires FP32 Q/K/V")
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise TypeError("paged-prefix Q/K/V dtypes must match")
    if q.device.type != "cuda":
        raise ValueError("FP32 paged-prefix attention requires CUDA tensors")
    if k_pool.device != q.device or v_pool.device != q.device:
        raise ValueError("paged-prefix Q/K/V tensors must be colocated")
    if (
        page_table.device != q.device
        or prefix_lengths.device != q.device
        or cu_seqlens_q.device != q.device
    ):
        raise ValueError("paged-prefix metadata must be colocated with Q/K/V")
    if k_pool.shape != v_pool.shape:
        raise ValueError("paged-prefix K/V shapes differ")
    if q.shape[-1] != k_pool.shape[-1]:
        raise ValueError("paged-prefix Q/K head dimensions differ")
    if page_table.shape[0] != prefix_lengths.shape[0]:
        raise ValueError("page table and prefix batch sizes differ")
    if cu_seqlens_q.shape != (prefix_lengths.shape[0] + 1,):
        raise ValueError("packed query offsets do not match the prefix batch")

    batch_size = prefix_lengths.shape[0]
    token_count, query_heads, head_dim = q.shape
    kv_heads = k_pool.shape[1]
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by K/V heads")

    page_indexes = page_table.long().clamp(0, k_pool.shape[0] - 1)
    key = k_pool[page_indexes]
    value = v_pool[page_indexes]

    token_indexes = torch.arange(token_count, device=q.device)
    batch_indexes = torch.bucketize(
        token_indexes,
        cu_seqlens_q[1:-1],
        right=True,
    )
    query_indexes = token_indexes - cu_seqlens_q[batch_indexes].long()
    padded_q = torch.zeros(
        (batch_size, max_q_len, query_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    padded_q[batch_indexes, query_indexes] = q

    queries_per_kv_head = query_heads // kv_heads
    grouped_q = padded_q.view(
        batch_size,
        max_q_len,
        kv_heads,
        queries_per_kv_head,
        head_dim,
    )
    scores = torch.einsum("bqhgd,bkhd->bqhgk", grouped_q, key)
    scores.mul_(head_dim**-0.5)
    valid_prefix = torch.arange(page_table.shape[1], device=q.device)[None, :] < (
        prefix_lengths[:, None]
    )
    scores.masked_fill_(~valid_prefix[:, None, None, None, :], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1).nan_to_num_(0.0)
    output = torch.einsum("bqhgk,bkhd->bqhgd", probabilities, value)
    output = output.reshape(batch_size, max_q_len, query_heads, head_dim)
    lse = lse.reshape(batch_size, max_q_len, query_heads)
    return (
        output[batch_indexes, query_indexes],
        lse[batch_indexes, query_indexes],
    )


def merge_attention_states_fp32(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> torch.Tensor:
    merged_lse = torch.logaddexp(prefix_lse, suffix_lse)
    output = (
        torch.exp(prefix_lse - merged_lse).unsqueeze(-1) * prefix_output
        + torch.exp(suffix_lse - merged_lse).unsqueeze(-1) * suffix_output
    )
    return output.to(prefix_output.dtype)
