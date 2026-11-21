"""Build MLA layer configs and layer inputs from the operands MLA models pass to the adapters."""

from typing import Tuple

import torch

from lightllm.common.attention.backends.mla import (
    MLADecodeLayerConfig,
    MLADecodeLayerInputs,
    MLAPrefillLayerConfig,
    MLAPrefillLayerInputs,
)
from .base_att import AttControl


def _check_unsupported_controls(att_control: AttControl) -> None:
    assert not (
        att_control.use_alibi or att_control.use_sliding_window or att_control.use_att_sink
    ), "MLA supports neither ALiBi, sliding windows nor sinks"


def mla_prefill_layer(
    q: torch.Tensor, k: Tuple[torch.Tensor, torch.Tensor], v: torch.Tensor, att_control: AttControl, causal: bool
) -> Tuple[MLAPrefillLayerInputs, MLAPrefillLayerConfig]:
    """k is (k_nope, k_rope); the layer's softmax scale comes in att_control.mla_prefill_dict."""
    _check_unsupported_controls(att_control)
    assert att_control.mla_prefill
    k_nope, k_rope = k
    layer_config = MLAPrefillLayerConfig(
        num_heads=q.shape[1],
        qk_nope_head_dim=k_nope.shape[-1],
        qk_rope_head_dim=k_rope.shape[-1],
        v_head_dim=v.shape[-1],
        softmax_scale=att_control.mla_prefill_dict["softmax_scale"],
        dtype=q.dtype,
        causal=causal,
    )
    return MLAPrefillLayerInputs(q=q, k_nope=k_nope, k_rope=k_rope, v=v), layer_config


def mla_decode_layer(
    q: Tuple[torch.Tensor, torch.Tensor], k: torch.Tensor, v, att_control: AttControl, causal: bool
) -> Tuple[MLADecodeLayerInputs, MLADecodeLayerConfig]:
    """q is (q_nope, q_rope) after absorption, k the latent cache; the softmax scale comes in mla_decode_dict."""
    _check_unsupported_controls(att_control)
    assert v is None and att_control.mla_decode
    q_nope, q_rope = q
    layer_config = MLADecodeLayerConfig(
        num_heads=q_nope.shape[1],
        kv_lora_rank=q_nope.shape[-1],
        qk_rope_head_dim=q_rope.shape[-1],
        softmax_scale=att_control.mla_decode_dict["softmax_scale"],
        dtype=q_nope.dtype,
        causal=causal,
    )
    return MLADecodeLayerInputs(q_nope=q_nope, q_rope=q_rope, kv_cache=k), layer_config
