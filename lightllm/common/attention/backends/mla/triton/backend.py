import dataclasses
from typing import Tuple

import torch

from lightllm.common.attention.api.backend import LayerBinding
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends.base import BaseAttentionBackend, PreparedPlan


@dataclasses.dataclass
class MlaTritonPrefillPlan(PreparedPlan):
    pass


@dataclasses.dataclass
class MlaTritonDecodePlan(PreparedPlan):
    block_seq: torch.Tensor = None
    batch_start: torch.Tensor = None
    indices_initialized: bool = False


class MlaTritonAttBackend(BaseAttentionBackend):
    provider = "triton"
    cache_formats = frozenset({"None"})
    family = AttentionFamily.MLA

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return MlaTritonPrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return MlaTritonDecodePlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        pass

    def _forward_prefill(self, layer: LayerBinding, plan, inputs: MLAPrefillInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        if q.shape[2] == 256:
            return self._prefill_contiguous_kv_prefill_att(
                layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func
            )
        else:
            return self._prefill_mla_prefill_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _prefill_mla_prefill_att(
        self,
        layer,
        plan,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        inputs,
        alloc_func=torch.empty,
    ):
        from lightllm.common.attention.backends.mla.triton.kernels.prefill import context_attention_fwd_with_v

        qk_rope_head_dim = layer.params.qk_rope_head_dim
        q_nope, q_rope = (q[:, :, :-qk_rope_head_dim], q[:, :, -qk_rope_head_dim:])
        o_tensor = alloc_func((q_nope.shape[0], q_nope.shape[1], v.shape[-1]), dtype=q_nope.dtype, device=q.device)
        k_nope, k_rope = k
        softmax_scale = layer.params.softmax_scale
        context_attention_fwd_with_v(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            o_tensor,
            plan.batch.b_q_start_loc,
            plan.batch.b1_cu_kv_seq_len,
            plan.batch.b_seq_len,
            plan.batch.b_ready_cache_len,
            plan.batch.max_q_seq_len,
            softmax_scale,
        )
        return o_tensor

    def _prefill_contiguous_kv_prefill_att(
        self,
        layer,
        plan,
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
            plan.batch.b_q_start_loc,
            plan.batch.b1_cu_kv_seq_len,
            plan.batch.b_seq_len,
            plan.batch.max_q_seq_len,
            plan.batch.b_ready_cache_len,
        )
        return o_tensor

    def _update_decode(self, plan):
        plan.block_seq = torch.empty((1,), dtype=torch.int64, device=plan.batch.device)
        plan.batch_start = torch.empty((plan.batch.batch_size,), dtype=torch.int64, device=plan.batch.device)
        plan.indices_initialized = False

    def _forward_decode(self, layer: LayerBinding, plan, inputs: MLADecodeInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.params.kv
        v = None
        assert v is None
        return self._decode_mla_decode_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _decode_mla_decode_att(
        self,
        layer,
        plan,
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
            batch=plan.batch,
            softmax_scale=softmax_scale,
            alloc_tensor_func=alloc_func,
            block_seq=plan.block_seq,
            batch_start=plan.batch_start,
            initialize=not plan.indices_initialized,
        )
        plan.indices_initialized = True
        return out
