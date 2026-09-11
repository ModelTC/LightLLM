from __future__ import annotations

import torch

from lightllm.utils.sgl_utils import (
    flash_attn_with_kvcache,
    get_scheduler_metadata,
    merge_state_v2,
)


def prepare_paged_prefix_attention(
    k_pool: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    query_heads: int,
    max_q_len: int,
    max_prefix_len: int,
):
    """Plan FA3 once for all model layers in one forward pass."""

    if k_pool.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"paged-prefix attention does not support {k_pool.dtype}")
    if get_scheduler_metadata is None:
        raise RuntimeError("FP16/BF16 paged-prefix attention requires FA3")
    return get_scheduler_metadata(
        batch_size=prefix_lengths.numel(),
        max_seqlen_q=max_q_len,
        max_seqlen_k=max_prefix_len,
        num_heads=query_heads,
        num_heads_k=k_pool.shape[1],
        headdim=k_pool.shape[-1],
        cache_seqlens=prefix_lengths,
        qkv_dtype=k_pool.dtype,
        cu_seqlens_q=cu_seqlens_q,
        page_size=1,
        causal=False,
    )


def paged_prefix_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    max_q_len: int,
    scheduler_metadata=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend packed queries to immutable paged prefix K/V."""

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"paged-prefix attention does not support {q.dtype}")
    if flash_attn_with_kvcache is None:
        raise RuntimeError("FP16/BF16 paged-prefix attention requires FA3")
    if k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise TypeError("paged-prefix Q/K/V dtypes must match")
    result = flash_attn_with_kvcache(
        q=q.contiguous(),
        k_cache=k_pool.view(k_pool.shape[0], 1, k_pool.shape[1], k_pool.shape[2]),
        v_cache=v_pool.view(v_pool.shape[0], 1, v_pool.shape[1], v_pool.shape[2]),
        page_table=page_table,
        cache_seqlens=prefix_lengths,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_q_len,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )
    return result[0], result[1].transpose(0, 1).contiguous()


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> torch.Tensor:
    """Merge two independently normalized attention states."""

    if prefix_output.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"attention-state merge does not support {prefix_output.dtype}")
    if merge_state_v2 is None:
        raise RuntimeError("FP16/BF16 attention-state merge requires FA3")
    output, _ = merge_state_v2(
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    )
    return output
