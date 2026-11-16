"""Typed per-call parameters for ``prefill_att`` / ``decode_att``.

Each attention family has its own parameter type. A state class names the type it accepts in
``params_type`` and the base class rejects anything else, so a layer cannot silently pass options
that its backend ignores.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


@dataclass(frozen=True)
class DenseAttParams:
    """Standard softmax attention.

    ``sliding_window``, ``sink_weight`` and ``alibi`` are optional features: ``None`` means off,
    and a backend must list the feature in ``supported_features`` to accept a non-None value.
    """

    # (left, right) window size, as taken by the kernels.
    sliding_window: Optional[Tuple[int, int]] = None
    # Per-head attention sink logits.
    sink_weight: Optional[torch.Tensor] = None
    # Per-head ALiBi slopes of this TP rank.
    alibi: Optional[torch.Tensor] = None
    # 静态 FP8 标定按此逻辑层号查询 State 的物理行；其他后端可不使用。
    logical_layer_id: Optional[int] = None

    FEATURES = ("sliding_window", "sink_weight", "alibi")

    def enabled_features(self) -> Tuple[str, ...]:
        return tuple(name for name in self.FEATURES if getattr(self, name) is not None)


@dataclass(frozen=True)
class MlaAttParams:
    softmax_scale: float


@dataclass(frozen=True)
class NsaPrefillAttParams:
    topk_mem_indices: torch.Tensor
    softmax_scale: float
    kv_lora_rank: int
    # Only read by the fp8 KV backends, which attend over the not-yet-quantized prefill KV.
    topk_indices: Optional[torch.Tensor] = None
    prefill_cache_kv: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class NsaDecodeAttParams:
    topk_mem_indices: torch.Tensor
    softmax_scale: float
    kv_lora_rank: int
    qk_rope_head_dim: int


@dataclass(frozen=True)
class Dsv4NsaAttParams:
    """DeepSeek-V4 sparse attention over the sliding-window KV plus an optional compressed KV."""

    compress_ratio: int
    head_dim_v: int
    softmax_scale: float
    attn_sink: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    # Indices into the compressed KV; None for layers without one.
    extra_indices: Optional[torch.Tensor] = None
    extra_lengths: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class GdnAttParams:
    """Gated delta net linear attention. The backend takes the fused projection instead of q/k/v."""

    mixed_qkvzba: torch.Tensor
    layer_weight: Any
    layer_num: int


@dataclass(frozen=True)
class KdaAttParams:
    """KDA linear attention. The backend takes the fused projection and gates instead of q/k/v."""

    mixed_qkv: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    layer_weight: Any
    layer_num: int
