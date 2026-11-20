from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GQAInputs:
    """Borrowed operands for causal grouped-query attention (including MHA and MQA), with scale 1/sqrt(head_dim).

    q is [total_query_tokens, q_heads, head_dim] in prefill, packed in batch
    order, or [batch_size, q_heads, head_dim] in decode (one final-position
    query per request). q_heads must be a positive multiple of kv_heads.

    k_cache/v_cache are already populated per-layer physical cache views of
    shape [cache_slots, kv_heads, head_dim]. Each provider's batch metadata
    addresses their first dimension. The caller owns and populates the cache,
    including the current queries' KV; backends do not write q or K/V.

    sliding_window=(-1, -1) attends to the full causal prefix. (left, 0), with
    left >= 0, includes positions [max(0, query_pos-left), query_pos].

    Supported dtypes, layouts and sliding-window modes are provider-specific
    and documented on each backend's forward.
    """

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    sliding_window: tuple[int, int] = (-1, -1)
