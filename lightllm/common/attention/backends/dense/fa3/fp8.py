from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.fa3.backend import (
    Fa3DensePrefillBackend,
    Fa3DenseDecodeBackend,
    Fa3DecodeContext,
    Fa3PrefillContext,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(slots=True)
class Fp8Fa3PrefillContext(Fa3PrefillContext):
    pass


@dataclasses.dataclass(slots=True)
class Fp8Fa3DecodeContext(Fa3DecodeContext):
    pass


class Fa3DenseFp8Common:
    family = AttentionFamily.DENSE
    supported_features = frozenset(
        {
            AttentionFeature.MTP,
            AttentionFeature.DYNAMIC_VERIFY,
            AttentionFeature.SINK,
            AttentionFeature.NONCAUSAL,
            AttentionFeature.MTP_WINDOW,
        }
    )

    provider = "fa3"

    cache_encodings = frozenset({CacheEncoding.FP8_HEAD})

    @property
    def kernel_page_size(self) -> int:
        return 1


class Fa3DenseFp8PrefillBackend(Fa3DenseFp8Common, Fa3DensePrefillBackend):
    phase = AttentionPhase.PREFILL

    def _run_prefill(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
            q_per_head_fp8_quant,
            q_per_head_static_fp8_quant,
        )
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        q_head_num = q.shape[1]
        q_head_dim = q.shape[2]
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]
        cache_k = k.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        cache_v = v.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        batch_size = context.batch.b_seq_len.shape[0]
        q = q.reshape(q.shape[0], k_head_num, -1)
        if layer.resources.query_scale is None:
            q, q_scale = q_per_head_fp8_quant(
                q=q, seq_lens=context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1], b1_start_loc=context.cu_seqlens_q
            )
        else:
            static_q_scales = layer.resources.query_scale
            q = q_per_head_static_fp8_quant(q=q, scales=static_q_scales)
            q_scale = static_q_scales.view(1, k_head_num).expand(batch_size, k_head_num)
        o = flash_attn_with_kvcache(
            q=q.reshape(-1, q_head_num, q_head_dim),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=context.page_table,
            cache_seqlens=context.batch.b_seq_len,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k_new=context.cu_seqlens_k,
            max_seqlen_q=context.batch.max_q_seq_len,
            causal=context.causal,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale,
            k_descale=layer.resources.key_scale.view(1, -1).expand(batch_size, -1),
            v_descale=layer.resources.value_scale.view(1, -1).expand(batch_size, -1),
            return_softmax_lse=False,
        )
        return o

    def allocate_context(self, capacity, *, capture, workspace) -> Fp8Fa3PrefillContext:
        return Fp8Fa3PrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: Fp8Fa3PrefillContext):
        super().build_device_metadata(context)

    def forward(
        self, layer: LayerAttention, context: Fp8Fa3PrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._run_prefill(layer, context, q=q, k=k, v=v)


class Fa3DenseFp8DecodeBackend(Fa3DenseFp8Common, Fa3DenseDecodeBackend):
    phase = AttentionPhase.DECODE

    def _run_decode(self, layer, context, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
            q_per_head_fp8_quant,
            q_per_head_static_fp8_quant,
        )
        from lightllm.utils.sgl_utils import flash_attn_with_kvcache

        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]
        cache_k = k.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        cache_v = v.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        q_head_num = q.shape[1]
        att_batch_size = context.b_att_seq_len.shape[0]
        q = q.reshape(q.shape[0], k_head_num, -1)
        if layer.resources.query_scale is None:
            q, q_scale = q_per_head_fp8_quant(
                q=q, seq_lens=context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1], b1_start_loc=context.cu_seqlens_q
            )
        else:
            static_q_scales = layer.resources.query_scale
            q = q_per_head_static_fp8_quant(q=q, scales=static_q_scales)
            q_scale = static_q_scales.view(1, k_head_num).expand(att_batch_size, k_head_num)
        o = flash_attn_with_kvcache(
            q=q.reshape(-1, q_head_num, k_head_dim),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=context.page_table,
            cache_seqlens=context.b_att_seq_len,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k_new=context.cu_seqlens_k,
            max_seqlen_q=context.decode_max_q_seq_len,
            causal=context.causal,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale.view(att_batch_size, k_head_num),
            k_descale=layer.resources.key_scale.view(1, -1).expand(att_batch_size, -1),
            v_descale=layer.resources.value_scale.view(1, -1).expand(att_batch_size, -1),
            return_softmax_lse=False,
        )
        return o

    def allocate_context(self, capacity, *, capture, workspace) -> Fp8Fa3DecodeContext:
        return Fp8Fa3DecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: Fp8Fa3DecodeContext):
        super().build_device_metadata(context)

    def forward(
        self, layer: LayerAttention, context: Fp8Fa3DecodeContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._run_decode(layer, context, q=q, k=k, v=v)
