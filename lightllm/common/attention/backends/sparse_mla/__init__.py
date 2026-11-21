"""Sparse MLA: absorbed MLA attention over the latent cache, restricted to each query's top-k cache slots.

The indexer that selects the slots belongs to the model; this family takes its resolved indices.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .. import require_positive

KV_CACHE_FORMATS = ("bf16", "fp8_ds_mla", "fp8_group64_e8m0")


@dataclass(frozen=True)
class SparseMLALayerConfig:
    """Static configuration of sparse MLA layers over the compressed latent cache.

    Head counts are per rank. Queries carry kv_lora_rank latent plus qk_rope_head_dim rotary
    channels, like each cache row; the latent part supplies values. qk_rope_head_dim is 0 for NoPE
    models (GLM-5). kv_cache_format is "bf16" (one [kv_lora_rank + qk_rope_head_dim] row per slot),
    "fp8_ds_mla" (DeepSeek V3.2's packed FP8 rows with fp32 scales per 128 channels and a bf16
    rotary part) or "fp8_group64_e8m0" (GLM-5's packed NoPE rows with E8M0 scales per 64 channels
    followed by indexer fields).
    """

    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    softmax_scale: float
    dtype: torch.dtype
    kv_cache_format: str = "bf16"

    def __post_init__(self) -> None:
        require_positive(self, "num_heads", "kv_lora_rank")
        if self.qk_rope_head_dim < 0:
            raise ValueError(f"qk_rope_head_dim must be >= 0, got {self.qk_rope_head_dim}")
        if self.kv_cache_format not in KV_CACHE_FORMATS:
            raise ValueError(f"kv_cache_format must be one of {KV_CACHE_FORMATS}, got {self.kv_cache_format!r}")


@dataclass(frozen=True)
class SparseMLAPrefillLayerInputs:
    """One layer's prefill operands.

    q is [query_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]. topk_mem_indices is
    [query_tokens, topk] cache slots, -1 for unused entries. An "fp8_ds_mla" cache also needs the
    step's own keys in full precision: current_kv [new_tokens, 1, kv_lora_rank + qk_rope_head_dim]
    and topk_indices, the same selection as positions in the step's ragged key layout, in which
    q_kv_start_loc gives each query's request start ("fp8_group64_e8m0" only).
    """

    q: torch.Tensor
    kv_cache: torch.Tensor
    topk_mem_indices: torch.Tensor
    topk_indices: Optional[torch.Tensor] = None
    current_kv: Optional[torch.Tensor] = None
    q_kv_start_loc: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SparseMLADecodeLayerInputs:
    """One layer's decode operands: q_nope [rows, num_heads, kv_lora_rank], q_rope [rows, num_heads,
    qk_rope_head_dim], the cache and topk_mem_indices [rows, topk] cache slots (-1 unused)."""

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_cache: torch.Tensor
    topk_mem_indices: torch.Tensor
