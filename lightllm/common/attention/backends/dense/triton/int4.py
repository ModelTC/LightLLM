from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend


@dataclasses.dataclass(slots=True)
class Int4kvTritonPrefillContext(AttentionContext):
    b_kv_start_loc: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class Int4kvTritonDecodeContext(AttentionContext):
    decode_max_kv_seq_len: int = None


class TritonDenseInt4Common:
    family = AttentionFamily.DENSE
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    provider = "triton"

    cache_encodings = frozenset({CacheEncoding.INT4})


class TritonDenseInt4PrefillBackend(
    TritonDenseInt4Common, AttentionBackend[Int4kvTritonPrefillContext, DenseInputs, torch.Tensor]
):
    phase = AttentionPhase.PREFILL

    def _run_prefill(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> torch.Tensor:
        total_token_num = context.batch.total_token_num
        head_dim = k.shape[2] * 2
        k_dequant = self.alloc_tensor((total_token_num, k.shape[1], head_dim), dtype=q.dtype, device=q.device)
        v_dequant = self.alloc_tensor((total_token_num, v.shape[1], head_dim), dtype=q.dtype, device=q.device)
        o_tensor = self.alloc_tensor(q.shape, dtype=q.dtype, device=q.device)
        max_kv_seq_len = context.batch.max_kv_seq_len
        from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import dequantize_int4kv

        dequantize_int4kv(
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
            quant_group_size=self.spec.cache.quant_group_size,
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

    def allocate_context(self, capacity, *, capture, workspace) -> Int4kvTritonPrefillContext:
        return Int4kvTritonPrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: Int4kvTritonPrefillContext):
        context.b_kv_start_loc = (
            torch.cumsum(context.batch.b_seq_len, dim=0, dtype=context.batch.b_seq_len.dtype) - context.batch.b_seq_len
        )

    def forward(
        self, layer: LayerAttention, context: Int4kvTritonPrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        k = (k, layer.resources.key_scale)
        v = (v, layer.resources.value_scale)
        k, k_scale = k
        v, v_scale = v
        o = self._run_prefill(layer, context, q=q, k=k, k_scale=k_scale, v=v, v_scale=v_scale)
        return o


class TritonDenseInt4DecodeBackend(
    TritonDenseInt4Common, AttentionBackend[Int4kvTritonDecodeContext, DenseInputs, torch.Tensor]
):
    phase = AttentionPhase.DECODE

    def _run_decode(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> torch.Tensor:
        from lightllm.common.attention.backends.dense.triton.kernels.decode.int4kv.ppl_int4kv_flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        return token_decode_attention_flash_decoding(
            q=q,
            batch=context.batch,
            cache_k=k,
            cache_k_scale=k_scale,
            cache_v=v,
            cache_v_scale=v_scale,
            max_kv_seq_len=context.decode_max_kv_seq_len,
            alloc_tensor_func=self.alloc_tensor,
        )

    def allocate_context(self, capacity, *, capture, workspace) -> Int4kvTritonDecodeContext:
        return Int4kvTritonDecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: Int4kvTritonDecodeContext):
        context.decode_max_kv_seq_len = context.batch.max_kv_seq_len

    def forward(
        self, layer: LayerAttention, context: Int4kvTritonDecodeContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        k = (k, layer.resources.key_scale)
        v = (v, layer.resources.value_scale)
        k, k_scale = k
        v, v_scale = v
        return self._run_decode(layer, context, q=q, k=k, k_scale=k_scale, v=v, v_scale=v_scale)
