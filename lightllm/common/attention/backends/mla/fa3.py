"""FA3 MLA. Decode reuses the FA3 GQA decode step builder: its row grouping and page table are the same."""

from dataclasses import dataclass

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBuildResources,
    TensorAllocator,
)
from lightllm.common.basemodel.infer_struct import InferStateInfo
from ..gqa.fa3 import FA3GQADecodeAttState, FA3GQADecodeBuildContext, FA3PageTables, fa3_requires_cpu_cache_sync
from . import MLADecodeLayerConfig, MLADecodeLayerInputs, MLAPrefillLayerConfig, MLAPrefillLayerInputs


@dataclass(frozen=True)
class FA3MLAPrefillAttState:
    """int32 cumulative query and kv offsets of the step's packed, expanded keys and values."""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


class FA3MLAPrefillBackend(AttentionBackend[MLAPrefillLayerConfig, FA3MLAPrefillAttState, MLAPrefillLayerInputs]):
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

        layer_config = self.layer_config
        q, k_nope = layer_inputs.q, layer_inputs.k_nope
        nope_dim, heads = layer_config.qk_nope_head_dim, layer_config.num_heads
        k = alloc_func((k_nope.shape[0], heads, q.shape[-1]), dtype=q.dtype, device=q.device)
        k[:, :, :nope_dim].copy_(k_nope)
        k[:, :, nope_dim:].copy_(layer_inputs.k_rope.expand(-1, heads, -1))
        flash_attn_varlen_func(
            q=q,
            k=k,
            v=layer_inputs.v,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            max_seqlen_k=att_state.max_seqlen_k,
            softmax_scale=layer_config.softmax_scale,
            causal=layer_config.causal,
            return_softmax_lse=False,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_mla_prefill"

    @classmethod
    def build(cls, metadata: InferStateInfo, context: None = None) -> FA3MLAPrefillAttState:
        assert (
            metadata.b1_cu_q_seq_len is not None and metadata.b1_cu_kv_seq_len is not None
        ), "prefill metadata fields required"
        return FA3MLAPrefillAttState(
            cu_seqlens_q=metadata.b1_cu_q_seq_len.int(),
            cu_seqlens_k=metadata.b1_cu_kv_seq_len.int(),
            max_seqlen_q=metadata.max_q_seq_len,
            max_seqlen_k=metadata.max_kv_seq_len,
        )

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return fa3_requires_cpu_cache_sync()


class FA3MLADecodeBackend(AttentionBackend[MLADecodeLayerConfig, FA3GQADecodeAttState, MLADecodeLayerInputs]):
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

        layer_config = self.layer_config
        kv, rank, page_size = layer_inputs.kv_cache, layer_config.kv_lora_rank, att_state.page_size
        flash_attn_with_kvcache(
            q=layer_inputs.q_rope,
            k_cache=kv[:, :, rank:].view(-1, page_size, 1, layer_config.qk_rope_head_dim),
            v_cache=kv[:, :, :rank].view(-1, page_size, 1, rank),
            qv=layer_inputs.q_nope,
            page_table=att_state.page_table,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k_new=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            softmax_scale=layer_config.softmax_scale,
            causal=layer_config.causal,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=False,
            out=out,
        )
        return out

    @staticmethod
    def get_name() -> str:
        return "fa3_mla_decode"

    build_context_type = FA3GQADecodeBuildContext

    @classmethod
    def build(cls, metadata: InferStateInfo, context: FA3GQADecodeBuildContext) -> FA3GQADecodeAttState:
        return context.prepare(metadata)

    @classmethod
    def create_build_context(cls, build_key: None, ctx: AttentionBuildResources) -> FA3GQADecodeBuildContext:
        # The latent cache is paged like an unquantized GQA cache.
        return FA3GQADecodeBuildContext(page_size=ctx.page_size, page_table_view=FA3PageTables(ctx, ctx.page_size).view)

    @classmethod
    def requires_cpu_cache_sync(cls) -> bool:
        return fa3_requires_cpu_cache_sync()
