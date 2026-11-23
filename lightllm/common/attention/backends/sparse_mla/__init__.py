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

    topk is the indexer's selection width, the number of cache slots a query reads at most; None
    when the layer's implementations do not need it. An indexer that scores pools of kpool keys
    selects whole pools within the top-k width plus the newest, partial pool. causal=False lets each
    query see all of its selected keys, as GLM-5's indexer already restricts them to visible keys.
    """

    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    softmax_scale: float
    dtype: torch.dtype
    kv_cache_format: str = "bf16"
    topk: Optional[int] = None
    kpool: int = 1
    causal: bool = True

    def __post_init__(self) -> None:
        require_positive(self, "num_heads", "kv_lora_rank", "kpool")
        if self.topk is not None:
            require_positive(self, "topk")
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
    fp8_workspace: Optional[dict[str, torch.Tensor]] = None


@dataclass(frozen=True)
class SparseMLADecodeLayerInputs:
    """One layer's decode operands: q_nope [rows, num_heads, kv_lora_rank], q_rope [rows, num_heads,
    qk_rope_head_dim], the cache and topk_mem_indices [rows, topk] cache slots (-1 unused)."""

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    kv_cache: torch.Tensor
    topk_mem_indices: torch.Tensor


@dataclass(frozen=True)
class SWASparseMLALayerConfig:
    """Static configuration of sparse MLA over a sliding-window cache plus an optional compressed cache
    (DeepSeek V4).

    num_heads is the per-rank head count padded to FlashMLA's 64-head tiles. Each query attends to
    its selected sliding-window rows and, when compress_ratio is not 0, to selected rows of the cache
    that compresses every compress_ratio tokens into one. has_attn_sink adds a learned sink logit per
    head.
    """

    num_heads: int
    head_dim_v: int
    softmax_scale: float
    dtype: torch.dtype
    compress_ratio: int = 0
    has_attn_sink: bool = False

    def __post_init__(self) -> None:
        require_positive(self, "num_heads", "head_dim_v")
        if self.compress_ratio < 0:
            raise ValueError(f"compress_ratio must be >= 0, got {self.compress_ratio}")


@dataclass(frozen=True)
class SWASparseMLALayerInputs:
    """One layer's operands: q [rows, num_heads, head_dim], FlashMLA attention views of the packed FP8
    sliding-window cache and of the compressed cache (None without compression), per-row selected
    indices with their valid lengths for each cache, and attn_sink [num_heads] when the layer has one."""

    q: torch.Tensor
    swa_cache: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    compressed_cache: Optional[torch.Tensor] = None
    extra_indices: Optional[torch.Tensor] = None
    extra_lengths: Optional[torch.Tensor] = None
    attn_sink: Optional[torch.Tensor] = None
