import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers


@dataclasses.dataclass(slots=True)
class TritonPrefillContext(AttentionExecutionContext):
    pass


@dataclasses.dataclass(slots=True)
class TritonDecodeContext(AttentionExecutionContext):
    b_mark_mtp_shared_group: torch.Tensor = None
    decode_max_kv_seq_len: int = None


class TritonAttBackend(BaseAttentionBackend[TritonPrefillContext | TritonDecodeContext, DenseInputs, torch.Tensor]):
    provider = "triton"
    supports_alibi = True
    supports_sliding_window = True
    supports_speculative_window = False
    cache_formats = frozenset({"None"})

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return TritonPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return TritonDecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        pass

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        if layer.params.alibi is not None:
            return self._prefill_alibi(layer, context, q=q, k=k, v=v, alloc_func=alloc_func)
        else:
            return self._prefill_paged(layer, context, q=q, k=k, v=v, alloc_func=alloc_func)

    def _prefill_alibi(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty):
        out = alloc_func(q.shape, q.dtype)
        from lightllm.common.basemodel.triton_kernel.alibi_att.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k,
            v,
            out,
            context.batch.b_req_idx,
            layer.params.alibi,
            context.batch.b_q_start_loc,
            context.batch.b_seq_len,
            context.batch.b_ready_cache_len,
            context.batch.max_q_seq_len,
            context.batch.req_to_token_indexs,
        )
        return out

    def _prefill_paged(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty):
        from lightllm.common.attention.backends._shared.prefill.context_flashattention_nopad import (
            context_attention_fwd,
        )

        if layer.spec.sliding_window is not None:
            sliding_window = layer.spec.sliding_window
        else:
            sliding_window = (-1, -1)
        out = alloc_func(q.shape, q.dtype)
        context_attention_fwd(
            q,
            k,
            v,
            out,
            context.batch.b_req_idx,
            context.batch.b_q_start_loc,
            context.batch.b_seq_len,
            context.batch.b_ready_cache_len,
            context.batch.max_q_seq_len,
            context.batch.req_to_token_indexs,
            sliding_window=sliding_window,
        )
        return out

    def _update_decode(self, context):
        context.decode_max_kv_seq_len = context.batch.max_kv_seq_len
        draft_step = self.decode_draft_step
        if draft_step > 0:
            context.b_mark_mtp_shared_group = build_mtp_shared_group_markers(
                context.batch.b_req_idx, hold_req_id=self.hold_req_id
            )

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        if layer.params.alibi is not None:
            return self._decode_alibi(layer, context, q=q, k=k, v=v, alloc_func=alloc_func)
        else:
            draft_step = self.decode_draft_step
            q_head_num = q.shape[1]
            k_head_num = k.shape[1]
            if draft_step > 0:
                assert q_head_num >= k_head_num, "speculative decode requires q_head_num >= k_head_num"
                return self._run_decode_speculative(layer, context, q=q, k=k, v=v, alloc_func=alloc_func)
            elif q_head_num == k_head_num and layer.spec.sliding_window is None:
                return self._decode_mha(layer, context, q=q, k=k, v=v, alloc_func=alloc_func)
            elif q_head_num >= k_head_num:
                return self._decode_gqa(layer, context, q=q, k=k, v=v, alloc_func=alloc_func)
            else:
                raise NotImplementedError("error")

    def _decode_alibi(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty):
        from lightllm.common.basemodel.triton_kernel.alibi_att.token_flashattention_nopad import token_attention_fwd

        out = alloc_func(q.shape, q.dtype)
        token_attention_fwd(
            q,
            k,
            v,
            out,
            layer.params.alibi,
            context.batch.req_to_token_indexs,
            context.batch.b_req_idx,
            context.batch.b_kv_start_loc,
            context.batch.b_seq_len,
            context.batch.max_kv_seq_len,
            context.batch.total_token_num,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _decode_mha(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.mha.flash.flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        out = alloc_func(q.shape, q.dtype)
        token_decode_attention_flash_decoding(
            q=q, batch=context.batch, cache_k=k, cache_v=v, out=out, alloc_tensor_func=alloc_func
        )
        return out

    def _decode_gqa(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.flash.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        if layer.spec.sliding_window is not None:
            sliding_window = layer.spec.sliding_window
        else:
            sliding_window = (-1, -1)
        out = alloc_func(q.shape, q.dtype)
        gqa_token_decode_attention_flash_decoding(
            q=q,
            batch=context.batch,
            cache_k=k,
            cache_v=v,
            max_len_in_batch=context.decode_max_kv_seq_len,
            out=out,
            alloc_tensor_func=alloc_func,
            sliding_window=sliding_window,
        )
        return out

    def _run_decode_speculative(
        self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alloc_func=torch.empty
    ):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.mtp_diverse import (
            token_decode_attention_mtp_diverse_single_token,
        )

        out = token_decode_attention_mtp_diverse_single_token(
            q=q,
            k=k,
            v=v,
            Req_to_tokens=context.batch.req_to_token_indexs,
            B_req_idx=context.batch.b_req_idx,
            b_seq_len=context.batch.b_seq_len,
            b_mark_shared_group=context.b_mark_mtp_shared_group,
            max_kv_len=context.decode_max_kv_seq_len,
            alloc_tensor_func=alloc_func,
        )
        return out
