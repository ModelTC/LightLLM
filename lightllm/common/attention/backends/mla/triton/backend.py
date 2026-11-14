from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses
from typing import Tuple

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend


@dataclasses.dataclass(slots=True)
class MlaTritonPrefillContext(AttentionContext):
    pass


@dataclasses.dataclass(slots=True)
class MlaTritonDecodeContext(AttentionContext):
    block_seq: torch.Tensor = None
    batch_start: torch.Tensor = None
    indices_initialized: bool = False


class TritonMLACommon:
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    provider = "triton"

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    family = AttentionFamily.MLA


class TritonMLAPrefillBackend(
    TritonMLACommon, AttentionBackend[MlaTritonPrefillContext, MLAPrefillInputs, torch.Tensor]
):
    phase = AttentionPhase.PREFILL

    def _run_prefill(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        inputs,
    ):
        from lightllm.common.attention.backends.mla.triton.kernels.prefill import context_attention_fwd_with_v

        qk_rope_head_dim = layer.resources.qk_rope_head_dim
        q_nope, q_rope = (q[:, :, :-qk_rope_head_dim], q[:, :, -qk_rope_head_dim:])
        o_tensor = self.alloc_tensor(
            (q_nope.shape[0], q_nope.shape[1], v.shape[-1]), dtype=q_nope.dtype, device=q_nope.device
        )
        k_nope, k_rope = k
        softmax_scale = self.spec.softmax_scale
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

    def _run_prefill_contiguous(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        inputs,
    ):
        from lightllm.common.attention.backends._shared.prefill.context_flashattention_nopad import (
            context_attention_fwd_contiguous_kv,
        )

        k_nope, k_rope = k
        q_head_num = q.shape[1]
        k_merged = torch.cat([k_nope.expand(-1, q_head_num, -1), k_rope.expand(-1, q_head_num, -1)], dim=-1)
        o_tensor = self.alloc_tensor(q.shape, dtype=q.dtype, device=q.device)
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

    def allocate_context(self, capacity, *, capture, workspace) -> MlaTritonPrefillContext:
        return MlaTritonPrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: MlaTritonPrefillContext):
        pass

    def forward(
        self, layer: LayerAttention, context: MlaTritonPrefillContext, inputs: MLAPrefillInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        if q.shape[2] == 256:
            return self._run_prefill_contiguous(layer, context, q=q, k=k, v=v, inputs=inputs)
        else:
            return self._run_prefill(layer, context, q=q, k=k, v=v, inputs=inputs)


class TritonMLADecodeBackend(TritonMLACommon, AttentionBackend[MlaTritonDecodeContext, MLADecodeInputs, torch.Tensor]):
    phase = AttentionPhase.DECODE

    def _run_decode(
        self,
        layer,
        context,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
    ):
        softmax_scale = self.spec.softmax_scale
        from lightllm.common.attention.backends.mla.triton.kernels.decode import (
            gqa_token_decode_attention_flash_decoding,
        )

        qk_rope_head_dim = layer.resources.qk_rope_head_dim
        q_nope, q_rope = q
        kv = k
        out = gqa_token_decode_attention_flash_decoding(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_nope=kv[:, :, :-qk_rope_head_dim],
            kv_rope=kv[:, :, -qk_rope_head_dim:],
            batch=context.batch,
            softmax_scale=softmax_scale,
            alloc_tensor_func=self.alloc_tensor,
            block_seq=context.block_seq,
            batch_start=context.batch_start,
            initialize=not context.indices_initialized,
        )
        context.indices_initialized = True
        return out

    def allocate_context(self, capacity, *, capture, workspace) -> MlaTritonDecodeContext:
        return MlaTritonDecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: MlaTritonDecodeContext):
        context.block_seq = torch.empty((1,), dtype=torch.int64, device=context.batch.device)
        context.batch_start = torch.empty((context.batch.batch_size,), dtype=torch.int64, device=context.batch.device)
        context.indices_initialized = False

    def forward(
        self, layer: LayerAttention, context: MlaTritonDecodeContext, inputs: MLADecodeInputs, *, out=None
    ) -> torch.Tensor:
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.resources.kv
        return self._run_decode(layer, context, q=q, k=k)
