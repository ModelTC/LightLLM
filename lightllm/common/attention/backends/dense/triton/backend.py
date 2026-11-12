from lightllm.common.attention.backends.base import BaseAttentionBackend
import dataclasses
import torch
from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers
from lightllm.common.attention.backends.base import PreparedPlan
from lightllm.common.attention.api.backend import LayerBinding
from lightllm.common.attention.api.inputs import DenseInputs


@dataclasses.dataclass
class TritonPrefillPlan(PreparedPlan):
    pass


@dataclasses.dataclass
class TritonDecodePlan(PreparedPlan):
    b_mark_mtp_shared_group: torch.Tensor = None
    decode_max_kv_seq_len: int = None


class TritonAttBackend(BaseAttentionBackend):
    provider = "triton"
    supports_alibi = True
    supports_sliding_window = True
    supports_speculative_window = False
    cache_formats = frozenset({"None"})

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return TritonPrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return TritonDecodePlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        pass

    def _forward_prefill(self, layer: LayerBinding, plan, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        if layer.params.alibi is not None:
            return self._prefill_alibi_prefill_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)
        else:
            return self._prefill_nomarl_prefill_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)

    def _prefill_alibi_prefill_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs, alloc_func=torch.empty
    ):
        out = alloc_func(q.shape, q.dtype)
        from lightllm.common.basemodel.triton_kernel.alibi_att.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k,
            v,
            out,
            plan.batch.b_req_idx,
            layer.params.alibi,
            plan.batch.b_q_start_loc,
            plan.batch.b_seq_len,
            plan.batch.b_ready_cache_len,
            plan.batch.max_q_seq_len,
            plan.batch.req_to_token_indexs,
        )
        return out

    def _prefill_nomarl_prefill_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs=None, alloc_func=torch.empty
    ):
        from lightllm.common.attention.backends._shared.prefill.context_flashattention_nopad import (
            context_attention_fwd,
        )

        if layer.params.sliding_window is not None:
            sliding_window = layer.params.sliding_window
        else:
            sliding_window = (-1, -1)
        out = alloc_func(q.shape, q.dtype)
        context_attention_fwd(
            q,
            k,
            v,
            out,
            plan.batch.b_req_idx,
            plan.batch.b_q_start_loc,
            plan.batch.b_seq_len,
            plan.batch.b_ready_cache_len,
            plan.batch.max_q_seq_len,
            plan.batch.req_to_token_indexs,
            sliding_window=sliding_window,
        )
        return out

    def _update_decode(self, plan):
        plan.decode_max_kv_seq_len = plan.batch.max_kv_seq_len
        draft_step = self.decode_draft_step
        if draft_step > 0:
            plan.b_mark_mtp_shared_group = build_mtp_shared_group_markers(
                plan.batch.b_req_idx, hold_req_id=self.hold_req_id
            )

    def _forward_decode(self, layer: LayerBinding, plan, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        if layer.params.alibi is not None:
            return self._decode_alibi_decode_att(layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func)
        else:
            draft_step = self.decode_draft_step
            q_head_num = q.shape[1]
            k_head_num = k.shape[1]
            if draft_step > 0:
                assert q_head_num >= k_head_num, "speculative decode requires q_head_num >= k_head_num"
                return self._decode_spec_decode_gqa_att(layer, plan, q=q, k=k, v=v, alloc_func=alloc_func)
            elif q_head_num == k_head_num and layer.params.sliding_window is None:
                return self._decode_normal_decode_flash_decoding_att(layer, plan, q=q, k=k, v=v, alloc_func=alloc_func)
            elif q_head_num >= k_head_num:
                return self._decode_normal_decode_gqa_flash_decoding_att(
                    layer, plan, q=q, k=k, v=v, inputs=inputs, alloc_func=alloc_func
                )
            else:
                raise NotImplementedError("error")

    def _decode_alibi_decode_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs, alloc_func=torch.empty
    ):
        from lightllm.common.basemodel.triton_kernel.alibi_att.token_flashattention_nopad import token_attention_fwd

        out = alloc_func(q.shape, q.dtype)
        token_attention_fwd(
            q,
            k,
            v,
            out,
            layer.params.alibi,
            plan.batch.req_to_token_indexs,
            plan.batch.b_req_idx,
            plan.batch.b_kv_start_loc,
            plan.batch.b_seq_len,
            plan.batch.max_kv_seq_len,
            plan.batch.total_token_num,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _decode_normal_decode_flash_decoding_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
    ):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.mha.flash.flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        out = alloc_func(q.shape, q.dtype)
        token_decode_attention_flash_decoding(
            q=q, batch=plan.batch, cache_k=k, cache_v=v, out=out, alloc_tensor_func=alloc_func
        )
        return out

    def _decode_normal_decode_gqa_flash_decoding_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs=None, alloc_func=torch.empty
    ):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.flash.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        if layer.params.sliding_window is not None:
            sliding_window = layer.params.sliding_window
        else:
            sliding_window = (-1, -1)
        out = alloc_func(q.shape, q.dtype)
        gqa_token_decode_attention_flash_decoding(
            q=q,
            batch=plan.batch,
            cache_k=k,
            cache_v=v,
            max_len_in_batch=plan.decode_max_kv_seq_len,
            out=out,
            alloc_tensor_func=alloc_func,
            sliding_window=sliding_window,
        )
        return out

    def _decode_spec_decode_gqa_att(
        self, layer, plan, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
    ):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.mtp_diverse import (
            token_decode_attention_mtp_diverse_single_token,
        )

        out = token_decode_attention_mtp_diverse_single_token(
            q=q,
            k=k,
            v=v,
            Req_to_tokens=plan.batch.req_to_token_indexs,
            B_req_idx=plan.batch.b_req_idx,
            b_seq_len=plan.batch.b_seq_len,
            b_mark_shared_group=plan.b_mark_mtp_shared_group,
            max_kv_len=plan.decode_max_kv_seq_len,
            alloc_tensor_func=alloc_func,
        )
        return out
