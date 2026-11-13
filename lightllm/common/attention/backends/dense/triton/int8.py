import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.basemodel.triton_kernel.diverse_utils import build_diverse_shared_group_markers
from lightllm.utils.envs_utils import enable_diverse_mode_gqa_decode_fast_kernel, get_env_start_args


@dataclasses.dataclass(slots=True)
class Int8kvTritonPrefillContext(AttentionExecutionContext):
    b_kv_start_loc: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class Int8kvTritonDecodeContext(AttentionExecutionContext):
    b_shared_seq_len: torch.Tensor = None
    b_mark_shared_group: torch.Tensor = None
    decode_max_kv_seq_len: int = None


class Int8kvTritonAttBackend(BaseAttentionBackend):
    provider = "triton"
    supports_alibi = False
    supports_sliding_window = False
    cache_formats = frozenset({"int8kv"})

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self.quant_group_size: int = get_env_start_args().llm_kv_quant_group_size

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return Int8kvTritonPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return Int8kvTritonDecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        context.b_kv_start_loc = (
            torch.cumsum(context.batch.b_seq_len, dim=0, dtype=context.batch.b_seq_len.dtype) - context.batch.b_seq_len
        )

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        k = (k, layer.params.key_scale)
        v = (v, layer.params.value_scale)
        k, k_scale = k
        v, v_scale = v
        o = self._run_prefill(layer, context, q=q, k=k, k_scale=k_scale, v=v, v_scale=v_scale, alloc_func=alloc_func)
        return o

    def _run_prefill(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        total_token_num = context.batch.total_token_num
        k_dequant = alloc_func((total_token_num, k.shape[1], k.shape[2]), dtype=q.dtype, device=q.device)
        v_dequant = alloc_func((total_token_num, v.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
        o_tensor = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        max_kv_seq_len = context.batch.max_kv_seq_len
        from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import dequantize_int8kv

        dequantize_int8kv(
            k=k,
            k_scale=k_scale,
            v=v,
            v_scale=v_scale,
            req_to_token_indexs=context.batch.req_to_token_indexs,
            b_seq_len=context.batch.b_seq_len,
            b_req_idx=context.batch.b_req_idx,
            b_kv_start_loc=context.b_kv_start_loc,
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
            b_start_loc=context.batch.b_q_start_loc,
            b_kv_start_loc=context.b_kv_start_loc,
            b_seq_len=context.batch.b_seq_len,
            max_q_input_len=context.batch.max_q_seq_len,
            b_prompt_cache_len=context.batch.b_ready_cache_len,
        )
        return o_tensor

    def _update_decode(self, context):
        context.decode_max_kv_seq_len = context.batch.max_kv_seq_len
        if enable_diverse_mode_gqa_decode_fast_kernel():
            context.b_mark_shared_group = build_diverse_shared_group_markers(
                b_shared_radix_node_id=context.batch.b_shared_radix_node_id
            )
            context.b_shared_seq_len = torch.where(context.b_mark_shared_group == 1, 0, context.batch.b_shared_seq_len)

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        k = (k, layer.params.key_scale)
        v = (v, layer.params.value_scale)
        k, k_scale = k
        v, v_scale = v
        if enable_diverse_mode_gqa_decode_fast_kernel():
            return self._run_decode_diverse(
                layer, context, q=q, k=k, k_scale=k_scale, v=v, v_scale=v_scale, alloc_func=alloc_func
            )
        else:
            return self._run_decode(
                layer, context, q=q, k=k, k_scale=k_scale, v=v, v_scale=v_scale, alloc_func=alloc_func
            )

    def _run_decode_diverse(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        from .kernels.decode.int8kv.int8kv_flash_decoding_diverse import token_decode_attention_flash_decoding

        return token_decode_attention_flash_decoding(
            q=q,
            batch=context.batch,
            cache_k=k,
            cache_k_scale=k_scale,
            cache_v=v,
            cache_v_scale=v_scale,
            alloc_tensor_func=alloc_func,
            b_shared_seq_len=context.b_shared_seq_len,
            b_mark_shared_group=context.b_mark_shared_group,
            shared_streams_dict=context.workspace.get_object("int8_decode_streams", dict),
        )

    def _run_decode(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
        alloc_func=torch.empty,
    ):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.int8kv.normal import (
            token_decode_attention_flash_decoding,
        )

        return token_decode_attention_flash_decoding(
            q=q,
            batch=context.batch,
            cache_k=k,
            cache_k_scale=k_scale,
            cache_v=v,
            cache_v_scale=v_scale,
            max_len_in_batch=context.decode_max_kv_seq_len,
            alloc_tensor_func=alloc_func,
        )
