import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend, Fa3DecodeContext, Fa3PrefillContext
from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
    q_per_head_fp8_quant,
    q_per_head_static_fp8_quant,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.sgl_utils import flash_attn_with_kvcache

logger = init_logger(__name__)


@dataclasses.dataclass
class Fp8Fa3PrefillContext(Fa3PrefillContext):
    pass


@dataclasses.dataclass
class Fp8Fa3DecodeContext(Fa3DecodeContext):
    pass


class Fp8Fa3AttBackend(Fa3AttBackend):
    provider = "fa3"
    supports_sliding_window = False
    cache_formats = frozenset({"fp8kv_sph"})

    def _init_infer_page_size(self):
        self.infer_page_size = 1
        assert (
            self.page_size % self.infer_page_size == 0
        ), f"model page_size {self.page_size} must be divisible by infer_page_size {self.infer_page_size}"
        logger.warning(
            f"Fp8Fa3AttBackend temporarily uses infer_page_size=1 with model page_size={self.page_size}; "
            "multi-token FP8 inference pages are not implemented yet."
        )

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return Fp8Fa3PrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return Fp8Fa3DecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        super()._update_prefill(context)

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._prefill_fp8_prefill_att(
            layer, context, q=q, k=k, v=v, alloc_func=alloc_func, logical_layer_id=layer.logical_layer_id
        )

    def _prefill_fp8_prefill_att(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logical_layer_id: int,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        q_head_num = q.shape[1]
        q_head_dim = q.shape[2]
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]
        cache_k = k.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        cache_v = v.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        batch_size = context.batch.b_seq_len.shape[0]
        q = q.reshape(q.shape[0], k_head_num, -1)
        if layer.params.query_scale is None:
            q, q_scale = q_per_head_fp8_quant(
                q=q, seq_lens=context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1], b1_start_loc=context.cu_seqlens_q
            )
        else:
            static_q_scales = layer.params.query_scale
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
            k_descale=layer.params.key_scale.view(1, -1).expand(batch_size, -1),
            v_descale=layer.params.value_scale.view(1, -1).expand(batch_size, -1),
            return_softmax_lse=False,
        )
        return o

    def _update_decode(self, context):
        super()._update_decode(context)

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._decode_fp8_decode_att(
            layer, context, q=q, k=k, v=v, alloc_func=alloc_func, logical_layer_id=layer.logical_layer_id
        )

    def _decode_fp8_decode_att(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logical_layer_id: int,
        alloc_func=torch.empty,
    ):
        k_head_num = k.shape[1]
        k_head_dim = k.shape[2]
        cache_k = k.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        cache_v = v.view(size=(-1, 1, k_head_num, k_head_dim)).view(torch.float8_e4m3fn)
        q_head_num = q.shape[1]
        att_batch_size = context.b_att_seq_len.shape[0]
        q = q.reshape(q.shape[0], k_head_num, -1)
        if layer.params.query_scale is None:
            q, q_scale = q_per_head_fp8_quant(
                q=q, seq_lens=context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1], b1_start_loc=context.cu_seqlens_q
            )
        else:
            static_q_scales = layer.params.query_scale
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
            k_descale=layer.params.key_scale.view(1, -1).expand(att_batch_size, -1),
            v_descale=layer.params.value_scale.view(1, -1).expand(att_batch_size, -1),
            return_softmax_lse=False,
        )
        return o
