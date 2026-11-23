import math
from dataclasses import dataclass
from typing import Optional, Union

import torch

from .. import require_positive


KV_QUANT_KINDS = (None, "fp8_per_tensor", "fp8_per_head", "int8_group", "int4_group")


@dataclass(frozen=True)
class GQAKVScales:
    """One layer's dequantization scales for a quantized KV cache (see GQALayerConfig.kv_quant).

    - fp8_per_tensor: k_scale/v_scale are host floats.
    - fp8_per_head: k_scale/v_scale are float32 [num_kv_heads] tensors; q_scale is the
      [num_kv_heads] static query scale when the layer config has has_q_scale, else None.
    - int8_group/int4_group: k_scale/v_scale are [cache_slots, num_kv_heads,
      head_dim // kv_quant_group_size] tensors addressed like the cache.
    """

    k_scale: Union[float, torch.Tensor]
    v_scale: Union[float, torch.Tensor]
    q_scale: Optional[torch.Tensor] = None


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

    kv_quant names how the cache stores K/V; None stores them in dtype. The quantized kinds are
    "fp8_per_tensor" and "fp8_per_head" (FP8 E4M3 with one static scale per layer or per KV
    head) and "int8_group"/"int4_group" (integers with one dynamic scale per token, KV head and
    kv_quant_group_size channels). Scales come per layer in GQALayerInputs.kv_scales.
    has_q_scale applies to "fp8_per_head": True quantizes q with static per-KV-head scales,
    False quantizes it dynamically per request.

    has_alibi=True adds ALiBi position biases, one slope per query head, supplied per layer as
    GQALayerInputs.alibi_slopes.

    bidirectional_image_spans=True (prefill) lets each query inside an image span also see the rest
    of its span, as given per step by AttentionMetadata.b_image_token_end (and b_image_token_start with a
    sliding window, whose span keys stay visible even outside the window); text queries stay causal.

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
    kv_quant: Optional[str] = None
    kv_quant_group_size: Optional[int] = None
    has_q_scale: bool = False
    has_alibi: bool = False
    bidirectional_image_spans: bool = False

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
        if self.kv_quant not in KV_QUANT_KINDS:
            raise ValueError(f"kv_quant must be one of {KV_QUANT_KINDS}, got {self.kv_quant!r}")
        grouped = self.kv_quant in ("int8_group", "int4_group")
        if grouped != (self.kv_quant_group_size is not None):
            raise ValueError("kv_quant_group_size must be given exactly for the int8/int4 group kinds")
        if grouped and (self.kv_quant_group_size <= 0 or self.head_dim % self.kv_quant_group_size):
            raise ValueError(f"kv_quant_group_size must divide head_dim, got {self.kv_quant_group_size}")
        if self.has_q_scale and self.kv_quant != "fp8_per_head":
            raise ValueError("has_q_scale applies only to kv_quant='fp8_per_head'")

    @property
    def has_default_scale(self) -> bool:
        return math.isclose(self.softmax_scale, 1.0 / math.sqrt(self.head_dim), rel_tol=1e-6)


@dataclass(frozen=True)
class GQALayerInputs:
    """Borrowed per-layer operands for grouped-query attention.

    q is [total_query_tokens, num_heads, head_dim] in prefill, packed in batch order, or
    [decode_rows, num_heads, head_dim] in decode, one row per batch row (see
    AttentionMetadata.decode_rows_per_request).

    k_cache/v_cache are already populated per-layer physical cache views of shape
    [cache_slots, num_kv_heads, head_dim], stored as the layer config's kv_quant: dtype, FP8 E4M3
    (or its uint8 bytes), int8, or int8 bytes packing two int4 values ([..., head_dim // 2]).
    Each implementation's attention state address their first dimension. The caller owns and populates
    the cache, including the current queries' KV; impls do not write q or K/V.

    sinks is the layer's [num_heads] sink logits in q's dtype when the layer_config has has_sinks, else None.
    kv_scales holds the cache's scales when the layer_config has kv_quant, else None.
    alibi_slopes is the layer's float32 [num_heads] ALiBi slopes when the layer_config has has_alibi, else None.
    """

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    sinks: Optional[torch.Tensor] = None
    kv_scales: Optional[GQAKVScales] = None
    alibi_slopes: Optional[torch.Tensor] = None
