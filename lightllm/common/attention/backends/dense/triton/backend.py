from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend


@dataclasses.dataclass(slots=True)
class TritonPrefillContext(AttentionContext):
    pass


@dataclasses.dataclass(slots=True)
class TritonDecodeContext(AttentionContext):
    b_mark_mtp_shared_group: torch.Tensor = None
    decode_max_kv_seq_len: int = None


class TritonDenseCommon:
    family = AttentionFamily.DENSE
    supported_features = frozenset(
        {AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.ALIBI, AttentionFeature.SLIDING_WINDOW}
    )

    provider = "triton"

    cache_encodings = frozenset({CacheEncoding.PLAIN})


class TritonDensePrefillBackend(TritonDenseCommon, AttentionBackend[TritonPrefillContext, DenseInputs, torch.Tensor]):
    phase = AttentionPhase.PREFILL

    def _prefill_alibi(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        out = self.alloc_tensor(q.shape, q.dtype)
        from lightllm.common.basemodel.triton_kernel.alibi_att.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k,
            v,
            out,
            context.batch.b_req_idx,
            layer.resources.alibi,
            context.batch.b_q_start_loc,
            context.batch.b_seq_len,
            context.batch.b_ready_cache_len,
            context.batch.max_q_seq_len,
            context.batch.req_to_token_indexs,
        )
        return out

    def _prefill_paged(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from lightllm.common.attention.backends._shared.prefill.context_flashattention_nopad import (
            context_attention_fwd,
        )

        if self.spec.sliding_window is not None:
            sliding_window = self.spec.sliding_window
        else:
            sliding_window = (-1, -1)
        out = self.alloc_tensor(q.shape, q.dtype)
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

    def allocate_context(self, capacity, *, capture, workspace) -> TritonPrefillContext:
        return TritonPrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: TritonPrefillContext):
        pass

    def forward(
        self, layer: LayerAttention, context: TritonPrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        if layer.resources.alibi is not None:
            return self._prefill_alibi(layer, context, q=q, k=k, v=v)
        else:
            return self._prefill_paged(layer, context, q=q, k=k, v=v)


class TritonDenseDecodeBackend(TritonDenseCommon, AttentionBackend[TritonDecodeContext, DenseInputs, torch.Tensor]):
    phase = AttentionPhase.DECODE

    def _decode_alibi(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.alibi_att.token_flashattention_nopad import token_attention_fwd

        out = self.alloc_tensor(q.shape, q.dtype)
        token_attention_fwd(
            q,
            k,
            v,
            out,
            layer.resources.alibi,
            context.batch.req_to_token_indexs,
            context.batch.b_req_idx,
            context.batch.b_kv_start_loc,
            context.batch.b_seq_len,
            context.batch.max_kv_seq_len,
            context.batch.total_token_num,
            alloc_tensor_func=self.alloc_tensor,
        )
        return out

    def _decode_mha(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.mha.flash.flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        out = self.alloc_tensor(q.shape, q.dtype)
        token_decode_attention_flash_decoding(
            q=q, batch=context.batch, cache_k=k, cache_v=v, out=out, alloc_tensor_func=self.alloc_tensor
        )
        return out

    def _decode_gqa(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from lightllm.common.attention.backends.dense.triton.kernels.decode.gqa.flash.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        if self.spec.sliding_window is not None:
            sliding_window = self.spec.sliding_window
        else:
            sliding_window = (-1, -1)
        out = self.alloc_tensor(q.shape, q.dtype)
        gqa_token_decode_attention_flash_decoding(
            q=q,
            batch=context.batch,
            cache_k=k,
            cache_v=v,
            max_len_in_batch=context.decode_max_kv_seq_len,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
            sliding_window=sliding_window,
        )
        return out

    def _run_decode_speculative(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
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
            alloc_tensor_func=self.alloc_tensor,
        )
        return out

    def allocate_context(self, capacity, *, capture, workspace) -> TritonDecodeContext:
        return TritonDecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: TritonDecodeContext):
        from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

        context.decode_max_kv_seq_len = context.batch.max_kv_seq_len
        draft_step = self.draft_step
        if draft_step > 0:
            context.b_mark_mtp_shared_group = build_mtp_shared_group_markers(
                context.batch.b_req_idx, hold_req_id=self.hold_req_id
            )

    def forward(
        self, layer: LayerAttention, context: TritonDecodeContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        if layer.resources.alibi is not None:
            return self._decode_alibi(layer, context, q=q, k=k, v=v)
        else:
            draft_step = self.draft_step
            q_head_num = q.shape[1]
            k_head_num = k.shape[1]
            if draft_step > 0:
                assert q_head_num >= k_head_num, "speculative decode requires q_head_num >= k_head_num"
                return self._run_decode_speculative(layer, context, q=q, k=k, v=v)
            elif q_head_num == k_head_num and self.spec.sliding_window is None:
                return self._decode_mha(layer, context, q=q, k=k, v=v)
            elif q_head_num >= k_head_num:
                return self._decode_gqa(layer, context, q=q, k=k, v=v)
            else:
                raise NotImplementedError("error")
