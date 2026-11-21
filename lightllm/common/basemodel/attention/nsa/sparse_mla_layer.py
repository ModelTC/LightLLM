"""Build sparse MLA layer configs and inputs from the NSA operands models pass to the adapters."""

from typing import Tuple

import torch

from lightllm.common.attention.backends.sparse_mla import (
    SparseMLADecodeLayerInputs,
    SparseMLALayerConfig,
    SparseMLAPrefillLayerInputs,
)
from ..base_att import AttControl


def sparse_mla_prefill_layer(
    q: torch.Tensor, kv_cache: torch.Tensor, att_control: AttControl, kv_cache_format: str
) -> Tuple[SparseMLAPrefillLayerInputs, SparseMLALayerConfig]:
    """q is the absorbed [tokens, heads, kv_lora_rank + rope] query; nsa_prefill_dict holds the indexer's
    selection, the layer's softmax scale and kv_lora_rank, and the step's own keys (prefill_cache_kv)."""
    assert att_control.nsa_prefill and att_control.nsa_prefill_dict is not None, "nsa_prefill_dict is required"
    nsa = att_control.nsa_prefill_dict
    rank = nsa["kv_lora_rank"]
    layer_config = SparseMLALayerConfig(
        num_heads=q.shape[1],
        kv_lora_rank=rank,
        qk_rope_head_dim=q.shape[-1] - rank,
        softmax_scale=nsa["softmax_scale"],
        dtype=q.dtype,
        kv_cache_format=kv_cache_format,
    )
    layer_inputs = SparseMLAPrefillLayerInputs(
        q=q,
        kv_cache=kv_cache,
        topk_mem_indices=nsa["topk_mem_indices"],
        topk_indices=nsa.get("topk_indices"),
        current_kv=nsa.get("prefill_cache_kv"),
    )
    return layer_inputs, layer_config


def sparse_mla_decode_layer(
    q: Tuple[torch.Tensor, torch.Tensor], kv_cache: torch.Tensor, att_control: AttControl, kv_cache_format: str
) -> Tuple[SparseMLADecodeLayerInputs, SparseMLALayerConfig]:
    """q is (q_nope, q_rope) after absorption; nsa_decode_dict holds the selection, scale and kv_lora_rank."""
    assert att_control.nsa_decode and att_control.nsa_decode_dict is not None, "nsa_decode_dict is required"
    nsa = att_control.nsa_decode_dict
    q_nope, q_rope = q
    layer_config = SparseMLALayerConfig(
        num_heads=q_nope.shape[1],
        kv_lora_rank=nsa["kv_lora_rank"],
        qk_rope_head_dim=q_rope.shape[-1],
        softmax_scale=nsa["softmax_scale"],
        dtype=q_nope.dtype,
        kv_cache_format=kv_cache_format,
    )
    layer_inputs = SparseMLADecodeLayerInputs(
        q_nope=q_nope, q_rope=q_rope, kv_cache=kv_cache, topk_mem_indices=nsa["topk_mem_indices"]
    )
    return layer_inputs, layer_config
