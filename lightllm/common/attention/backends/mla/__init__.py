from dataclasses import dataclass

import torch

from .. import require_positive


@dataclass(frozen=True)
class MLAPrefillLayerConfig:
    """Static configuration of MLA prefill over expanded (non-absorbed) keys and values.

    Head counts are per rank. Queries and keys have qk_nope_head_dim + qk_rope_head_dim
    channels; values and outputs have v_head_dim. Prefill and decode have separate layer configs
    because they see different projections of the same layer. causal=False lets the queries of
    one request see all of its keys, as block-drafting models need.
    """

    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    softmax_scale: float
    dtype: torch.dtype
    causal: bool = True

    def __post_init__(self) -> None:
        require_positive(self, "num_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")


@dataclass(frozen=True)
class MLAPrefillLayerInputs:
    """Per-layer projected keys/values in request order, including cached prefixes.

    q is [query_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]. k_nope is
    [kv_tokens, num_heads, qk_nope_head_dim], k_rope is [kv_tokens, 1, qk_rope_head_dim]
    and v is [kv_tokens, num_heads, v_head_dim].
    """

    q: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class MLADecodeLayerConfig:
    """Static configuration of MLA decode over the compressed latent cache.

    Head counts are per rank. The absorbed query has kv_lora_rank latent channels plus
    qk_rope_head_dim rotary channels, matching each cache row. causal applies when speculative
    decoding gives a request several query rows (see AttentionBatch.decode_rows_per_request).
    """

    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    softmax_scale: float
    dtype: torch.dtype
    causal: bool = True

    def __post_init__(self) -> None:
        require_positive(self, "num_heads", "kv_lora_rank", "qk_rope_head_dim")


@dataclass(frozen=True)
class MLADecodeLayerInputs:
    """Absorbed query rows, split into latent and rotary components.

    q_nope is [rows, num_heads, kv_lora_rank] and q_rope is [rows, num_heads,
    qk_rope_head_dim]. kv_cache is a borrowed [physical_slots, 1, kv_lora_rank +
    qk_rope_head_dim] cache whose latent prefix supplies both K and V. The cache includes
    the current token for each request.
    """

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_cache: torch.Tensor
