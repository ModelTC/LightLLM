import dataclasses
from typing import Tuple

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends.base import AttentionExecutionContext
from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend
from lightllm.utils.sgl_utils import flash_attn_varlen_func, flash_attn_with_kvcache


@dataclasses.dataclass(slots=True)
class MlaFa3PrefillContext(AttentionExecutionContext):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    causal: bool = None


@dataclasses.dataclass(slots=True)
class MlaFa3DecodeContext(AttentionExecutionContext):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None
    decode_max_kv_seq_len: int = None
    causal: bool = None


class MlaFa3AttBackend(Fa3AttBackend):
    supports_noncausal_prefill = True
    provider = "fa3"
    max_head_size = None
    cache_formats = frozenset({"None"})
    family = AttentionFamily.MLA

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return MlaFa3PrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return MlaFa3DecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        context.causal = self.uses_causal_attention()
        context.cu_seqlens_q = context.batch.b1_cu_q_seq_len.int()
        context.cu_seqlens_k = context.batch.b1_cu_kv_seq_len.int()

    def _forward_prefill(
        self, layer: LayerAttentionBinding, context, inputs: MLAPrefillInputs, *, out=None
    ) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        return self._run_prefill(layer, context, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _run_prefill(
        self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs, alloc_func=torch.empty
    ) -> torch.Tensor:
        k_nope, k_rope = k
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        assert q.ndim == 3 and k.ndim == 3 and (v.ndim == 3)
        softmax_scale = layer.spec.softmax_scale
        o_tensor = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.batch.max_q_seq_len,
            max_seqlen_k=context.batch.max_kv_seq_len,
            softmax_scale=softmax_scale,
            causal=context.causal,
            return_softmax_lse=False,
        )
        return o_tensor

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: MLADecodeInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.params.kv
        return self._run_decode(layer, context, q=q, k=k, alloc_func=alloc_func)

    def _run_decode(
        self,
        layer,
        context,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        alloc_func=torch.empty,
    ):
        q_nope, q_rope = q
        kv = k
        qk_rope_head_dim = layer.params.qk_rope_head_dim
        kv_lora_rank = kv.shape[-1] - qk_rope_head_dim
        k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, self.infer_page_size, 1, qk_rope_head_dim)
        kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, self.infer_page_size, 1, kv_lora_rank)
        k_descale, v_descale = (None, None)
        softmax_scale = layer.spec.softmax_scale
        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=context.page_table,
            cache_seqlens=context.b_att_seq_len,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k_new=context.cu_seqlens_k,
            max_seqlen_q=context.decode_max_q_seq_len,
            softmax_scale=softmax_scale,
            causal=context.causal,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
        )
        return o_tensor
