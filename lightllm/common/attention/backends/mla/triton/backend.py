import dataclasses
from typing import Tuple

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend


@dataclasses.dataclass(slots=True)
class MlaTritonPrefillContext(AttentionExecutionContext):
    pass


@dataclasses.dataclass(slots=True)
class MlaTritonDecodeContext(AttentionExecutionContext):
    block_seq: torch.Tensor = None
    batch_start: torch.Tensor = None
    indices_initialized: bool = False


class MlaTritonAttBackend(BaseAttentionBackend):
    provider = "triton"
    cache_formats = frozenset({"None"})
    family = AttentionFamily.MLA

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return MlaTritonPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return MlaTritonDecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        pass

    def _forward_prefill(
        self, layer: LayerAttentionBinding, context, inputs: MLAPrefillInputs, *, out=None
    ) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        if q.shape[2] == 256:
            return self._prefill_contiguous_kv_prefill_att(
                layer, context, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func
            )
        else:
            return self._prefill_mla_prefill_att(layer, context, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _prefill_mla_prefill_att(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        inputs,
        alloc_func=torch.empty,
    ):
        from lightllm.common.attention.backends.mla.triton.kernels.prefill import context_attention_fwd_with_v

        qk_rope_head_dim = layer.params.qk_rope_head_dim
        q_nope, q_rope = (q[:, :, :-qk_rope_head_dim], q[:, :, -qk_rope_head_dim:])
        o_tensor = alloc_func((q_nope.shape[0], q_nope.shape[1], v.shape[-1]), dtype=q_nope.dtype, device=q_nope.device)
        k_nope, k_rope = k
        softmax_scale = layer.params.softmax_scale
        context_attention_fwd_with_v(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            o_tensor,
            context.batch.b_q_start_loc,
            context.batch.b1_cu_kv_seq_len,
            context.batch.b_seq_len,
            context.batch.b_ready_cache_len,
            context.batch.max_q_seq_len,
            softmax_scale,
        )
        return o_tensor

    def _prefill_contiguous_kv_prefill_att(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        inputs,
        alloc_func=torch.empty,
    ):
        from lightllm.common.attention.backends._shared.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        k_nope, k_rope = k
        q_head_num = q.shape[1]
        k_merged = torch.cat([k_nope.expand(-1, q_head_num, -1), k_rope.expand(-1, q_head_num, -1)], dim=-1)
        o_tensor = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        context_attention_fwd_contiguous_kv(
            q,
            k_merged,
            v,
            o_tensor,
            context.batch.b_q_start_loc,
            context.batch.b1_cu_kv_seq_len,
            context.batch.b_seq_len,
            context.batch.max_q_seq_len,
            context.batch.b_ready_cache_len,
        )
        return o_tensor

    def _update_decode(self, context):
        context.block_seq = torch.empty((1,), dtype=torch.int64, device=context.batch.device)
        context.batch_start = torch.empty((context.batch.batch_size,), dtype=torch.int64, device=context.batch.device)
        context.indices_initialized = False

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: MLADecodeInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.params.kv
        v = None
        assert v is None
        return self._decode_mla_decode_att(layer, context, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _decode_mla_decode_att(
        self,
        layer,
        context,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        inputs,
        alloc_func=torch.empty,
    ):
        softmax_scale = layer.params.softmax_scale
        from lightllm.common.attention.backends.mla.triton.kernels.decode import (
            gqa_token_decode_attention_flash_decoding,
        )

        qk_rope_head_dim = layer.params.qk_rope_head_dim
        q_nope, q_rope = q
        kv = k
        out = gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=kv[:, :, :-qk_rope_head_dim],
            kv_rope=kv[:, :, -qk_rope_head_dim:],
            batch=context.batch,
            softmax_scale=softmax_scale,
            alloc_tensor_func=alloc_func,
            block_seq=context.block_seq,
            batch_start=context.batch_start,
            initialize=not context.indices_initialized,
        )
        context.indices_initialized = True
        return out
