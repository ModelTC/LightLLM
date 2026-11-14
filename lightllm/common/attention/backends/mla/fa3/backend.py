from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase
import dataclasses
from typing import Tuple

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import MLADecodeInputs, MLAPrefillInputs
from lightllm.common.attention.api.backend import AttentionContext
from lightllm.common.attention.backends.dense.fa3.backend import Fa3DensePrefillBackend, Fa3DenseDecodeBackend


@dataclasses.dataclass(slots=True)
class MlaFa3PrefillContext(AttentionContext):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    causal: bool = None


@dataclasses.dataclass(slots=True)
class MlaFa3DecodeContext(AttentionContext):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None
    decode_max_kv_seq_len: int = None
    causal: bool = None


class Fa3MLACommon:
    provider = "fa3"

    max_head_size = None

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    family = AttentionFamily.MLA


class Fa3MLAPrefillBackend(Fa3MLACommon, Fa3DensePrefillBackend):
    phase = AttentionPhase.PREFILL

    def _run_prefill(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inputs) -> torch.Tensor:
        from lightllm.utils.sgl_utils import flash_attn_varlen_func

        k_nope, k_rope = k
        q_head_num = q.shape[1]
        k = torch.cat([k_nope, torch.repeat_interleave(k_rope, q_head_num, dim=-2)], dim=-1)
        assert q.ndim == 3 and k.ndim == 3 and (v.ndim == 3)
        softmax_scale = self.spec.softmax_scale
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

    def allocate_context(self, capacity, *, capture, workspace) -> MlaFa3PrefillContext:
        return MlaFa3PrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: MlaFa3PrefillContext):
        context.causal = self.spec.causal
        context.cu_seqlens_q = context.batch.b1_cu_q_seq_len.int()
        context.cu_seqlens_k = context.batch.b1_cu_kv_seq_len.int()

    def forward(
        self, layer: LayerAttention, context: MlaFa3PrefillContext, inputs: MLAPrefillInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = (inputs.key_nope, inputs.key_rope)
        v = inputs.value
        return self._run_prefill(layer, context, q=q, k=k, v=v, inputs=inputs)


class Fa3MLADecodeBackend(Fa3MLACommon, Fa3DenseDecodeBackend):
    phase = AttentionPhase.DECODE

    def _run_decode(self, layer, context, q: Tuple[torch.Tensor, torch.Tensor], k: torch.Tensor):
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        q_nope, q_rope = q
        kv = k
        qk_rope_head_dim = layer.resources.qk_rope_head_dim
        kv_lora_rank = kv.shape[-1] - qk_rope_head_dim
        k_rope = kv[:, :, -qk_rope_head_dim:].view(-1, self.kernel_page_size, 1, qk_rope_head_dim)
        kv_nope = kv[:, :, :-qk_rope_head_dim].view(-1, self.kernel_page_size, 1, kv_lora_rank)
        k_descale, v_descale = (None, None)
        softmax_scale = self.spec.softmax_scale
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

    def allocate_context(self, capacity, *, capture, workspace) -> MlaFa3DecodeContext:
        return MlaFa3DecodeContext(capacity, capture, workspace)

    def forward(
        self, layer: LayerAttention, context: MlaFa3DecodeContext, inputs: MLADecodeInputs, *, out=None
    ) -> torch.Tensor:
        q = (inputs.query_nope, inputs.query_rope)
        k = layer.resources.kv
        return self._run_decode(layer, context, q=q, k=k)
