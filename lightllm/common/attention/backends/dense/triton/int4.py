import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.base import BaseAttentionBackend, PreparedPlan
from lightllm.utils.envs_utils import get_env_start_args


@dataclasses.dataclass
class Int4kvTritonPrefillPlan(PreparedPlan):
    b_kv_start_loc: torch.Tensor = None


@dataclasses.dataclass
class Int4kvTritonDecodePlan(PreparedPlan):
    decode_max_kv_seq_len: int = None


class Int4kvTritonAttBackend(BaseAttentionBackend):
    provider = "triton"
    supports_alibi = False
    supports_sliding_window = False
    cache_formats = frozenset({"int4kv"})

    def __init__(self, model):
        super().__init__(model)
        self.quant_group_size: int = get_env_start_args().llm_kv_quant_group_size

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return Int4kvTritonPrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return Int4kvTritonDecodePlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        plan.b_kv_start_loc = (
            torch.cumsum(plan.batch.b_seq_len, dim=0, dtype=plan.batch.b_seq_len.dtype) - plan.batch.b_seq_len
        )

    def _forward_prefill(self, layer: LayerBinding, plan, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        k = (k, layer.params.key_scale)
        v = (v, layer.params.value_scale)
        k, k_scale = k
        v, v_scale = v
        o = self._prefill_groupsize_quant_prefill_att(
            layer, plan, q=q, k=k, k_scale=k_scale, v=v, v_scale=v_scale, alloc_func=alloc_func
        )
        return o

    def _prefill_groupsize_quant_prefill_att(
        self,
        layer,
        plan,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        total_token_num = plan.batch.total_token_num
        head_dim = k.shape[2] * 2
        k_dequant = alloc_func((total_token_num, k.shape[1], head_dim), dtype=q.dtype, device=q.device)
        v_dequant = alloc_func((total_token_num, v.shape[1], head_dim), dtype=q.dtype, device=q.device)
        o_tensor = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        max_kv_seq_len = plan.batch.max_kv_seq_len
        from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import dequantize_int4kv

        dequantize_int4kv(
            k=k,
            k_scale=k_scale,
            v=v,
            v_scale=v_scale,
            req_to_token_indexs=plan.batch.req_to_token_indexs,
            b_seq_len=plan.batch.b_seq_len,
            b_req_idx=plan.batch.b_req_idx,
            b_kv_start_loc=plan.b_kv_start_loc,
            k_out=k_dequant,
            v_out=v_dequant,
            max_len_in_batch=max_kv_seq_len,
            quant_group_size=self.quant_group_size,
        )
        from lightllm.common.attention.backends._shared.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        context_attention_fwd_contiguous_kv(
            q=q,
            k=k_dequant,
            v=v_dequant,
            o=o_tensor,
            b_start_loc=plan.batch.b_q_start_loc,
            b_kv_start_loc=plan.b_kv_start_loc,
            b_seq_len=plan.batch.b_seq_len,
            max_q_input_len=plan.batch.max_q_seq_len,
            b_prompt_cache_len=plan.batch.b_ready_cache_len,
        )
        return o_tensor

    def _update_decode(self, plan):
        plan.decode_max_kv_seq_len = plan.batch.max_kv_seq_len

    def _forward_decode(self, layer: LayerBinding, plan, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        k = (k, layer.params.key_scale)
        v = (v, layer.params.value_scale)
        k, k_scale = k
        v, v_scale = v
        return self._decode_ppl_int4kv_decode_att(
            layer, plan, q=q, k=k, k_scale=k_scale, v=v, v_scale=v_scale, alloc_func=alloc_func
        )

    def _decode_ppl_int4kv_decode_att(
        self,
        layer,
        plan,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        from lightllm.common.attention.backends.dense.triton.kernels.decode.int4kv.ppl_int4kv_flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        return token_decode_attention_flash_decoding(
            q=q,
            batch=plan.batch,
            cache_k=k,
            cache_k_scale=k_scale,
            cache_v=v,
            cache_v_scale=v_scale,
            max_kv_seq_len=plan.decode_max_kv_seq_len,
            alloc_tensor_func=alloc_func,
        )
