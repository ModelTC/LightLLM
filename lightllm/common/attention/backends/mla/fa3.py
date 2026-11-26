"""FA3 MLA. Decode reuses the FA3 GQA decode state: its row grouping and page table are the same."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass

import torch

from lightllm.common.attention.backend import AttentionBackend, AttentionBuildResources, TensorAllocator
from ..gqa.fa3 import FA3GQADecodeAttState, build_fa3_decode_state, fa3_requires_cpu_cache_sync
from . import (
    MLADecodeLayerConfig,
    MLADecodeLayerInputs,
    MLAPrefillLayerConfig,
    MLAPrefillLayerInputs,
    expand_mla_keys,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


@dataclass(frozen=True)
class FA3MLAPrefillAttState:
    """int32 cumulative query and kv offsets of the step's packed, expanded keys and values."""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


class FA3MLAPrefillBackend(AttentionBackend[MLAPrefillLayerConfig, FA3MLAPrefillAttState, MLAPrefillLayerInputs]):
    name = "fa3_mla_prefill"
    phase = "prefill"

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return fa3_requires_cpu_cache_sync()

    def build(self, infer_state: InferStateInfo) -> FA3MLAPrefillAttState:
        return FA3MLAPrefillAttState(
            cu_seqlens_q=infer_state.b1_cu_q_seq_len.int(),
            cu_seqlens_k=infer_state.b1_cu_kv_seq_len.int(),
            max_seqlen_q=infer_state.max_q_seq_len,
            max_seqlen_k=infer_state.max_kv_seq_len,
        )

    def forward(
        self,
        att_state: FA3MLAPrefillAttState,
        layer_inputs: MLAPrefillLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [query_tokens, num_heads, v_head_dim]; keys are expanded into alloc_func scratch.

        FA3 aligns the causal mask to each request's last key, so cached prefixes stay visible.
        """
        from lightllm.utils.sgl_utils import flash_attn_varlen_func

        config = self.layer_config
        flash_attn_varlen_func(
            q=layer_inputs.q,
            k=expand_mla_keys(layer_inputs, alloc_func),
            v=layer_inputs.v,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            max_seqlen_k=att_state.max_seqlen_k,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
            return_softmax_lse=False,
            out=out,
        )
        return out


class FA3MLADecodeBackend(AttentionBackend[MLADecodeLayerConfig, FA3GQADecodeAttState, MLADecodeLayerInputs]):
    name = "fa3_mla_decode"
    phase = "decode"

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return fa3_requires_cpu_cache_sync()

    def setup(self, resources: AttentionBuildResources) -> None:
        # The latent cache is paged like an unquantized GQA cache.
        self.resources = resources
        self.page_size = resources.page_size

    def build(self, infer_state: InferStateInfo) -> FA3GQADecodeAttState:
        return build_fa3_decode_state(self, infer_state)

    def forward(
        self,
        att_state: FA3GQADecodeAttState,
        layer_inputs: MLADecodeLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's shape. The rotary part is FA3's q/k and the latent part its qv/v."""
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        config = self.layer_config
        kv, rank, page_size = layer_inputs.kv_cache, config.kv_lora_rank, att_state.page_size
        flash_attn_with_kvcache(
            q=layer_inputs.q_rope,
            k_cache=kv[:, :, rank:].view(-1, page_size, 1, config.qk_rope_head_dim),
            v_cache=kv[:, :, :rank].view(-1, page_size, 1, rank),
            qv=layer_inputs.q_nope,
            page_table=att_state.page_table,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k_new=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            out=out,
        )
        return out
