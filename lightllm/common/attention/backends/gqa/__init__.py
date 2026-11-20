import math
from dataclasses import dataclass
from typing import Optional

import torch

from .. import require_positive


@dataclass(frozen=True)
class GQALayerConfig:
    """Static configuration of grouped-query attention layers (including MHA and MQA).

    Head counts are per rank; num_heads must be a positive multiple of num_kv_heads.
    softmax_scale defaults to 1/sqrt(head_dim).

    causal=True limits each query to keys up to its own position; causal=False lets the queries
    of one request see all of its keys, as block-drafting models need. sliding_window=(-1, -1)
    attends to every visible key. (left, right), with both sides >= 0, limits keys to positions
    [query_pos - left, query_pos + right]; causal attention uses (left, 0), and only non-causal
    attention (block drafting) may look right. Any two-element sequence is accepted and stored as a tuple, so
    a list loaded from a JSON model config compares equal to the tuple forms above.

    has_sinks=True adds one learned sink logit per query head to every softmax, supplied per
    layer as GQALayerInputs.sinks.

    Supported dtypes, head dimensions, scales and sliding-window modes are documented on each impl.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    sliding_window: tuple[int, int] = (-1, -1)
    softmax_scale: Optional[float] = None
    causal: bool = True
    has_sinks: bool = False

    def __post_init__(self) -> None:
        require_positive(self, "num_heads", "num_kv_heads", "head_dim")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a positive multiple of num_kv_heads ({self.num_kv_heads})"
            )
        left, right = (int(x) for x in self.sliding_window)
        if (left, right) != (-1, -1) and (left < 0 or right < 0):
            raise ValueError(f"sliding_window must be (-1, -1) or have both sides >= 0, got {(left, right)}")
        if self.causal and right > 0:
            raise ValueError(f"a causal sliding_window cannot look right, got {(left, right)}")
        object.__setattr__(self, "sliding_window", (left, right))
        if self.softmax_scale is None:
            object.__setattr__(self, "softmax_scale", 1.0 / math.sqrt(self.head_dim))

    @property
    def has_default_scale(self) -> bool:
        return math.isclose(self.softmax_scale, 1.0 / math.sqrt(self.head_dim), rel_tol=1e-6)


@dataclass(frozen=True)
class GQALayerInputs:
    """Borrowed per-layer operands for grouped-query attention.

    q is [total_query_tokens, num_heads, head_dim] in prefill, packed in batch order, or
    [decode_rows, num_heads, head_dim] in decode, one row per batch row (see
    AttentionBatch.decode_rows_per_request).

    k_cache/v_cache are already populated per-layer physical cache views of shape
    [cache_slots, num_kv_heads, head_dim]. Each implementation's step args address their first
    dimension. The caller owns and populates the cache, including the current queries' KV;
    impls do not write q or K/V.

    sinks is the layer's [num_heads] sink logits in q's dtype when the layer_config has has_sinks, else None.
    """

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    sinks: Optional[torch.Tensor] = None
