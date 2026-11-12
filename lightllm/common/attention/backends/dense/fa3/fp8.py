import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend, Fa3DecodePlan, Fa3PrefillPlan
from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
    q_per_head_fp8_quant,
    q_per_head_static_fp8_quant,
)
from lightllm.utils.log_utils import init_logger
from lightllm.utils.sgl_utils import flash_attn_with_kvcache

logger = init_logger(__name__)


@dataclasses.dataclass
class Fp8Fa3PrefillPlan(Fa3PrefillPlan):
    pass


@dataclasses.dataclass
class Fp8Fa3DecodePlan(Fa3DecodePlan):
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

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return Fp8Fa3PrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return Fp8Fa3DecodePlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        super()._update_prefill(plan)

    def _forward_prefill(self, layer: LayerAttentionBinding, plan, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._prefill_fp8_prefill_att(
            layer, plan, q=q, k=k, v=v, alloc_func=alloc_func, logical_layer_id=layer.logical_layer_id
        )

    def _prefill_fp8_prefill_att(
        self,
        layer,
        plan,
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
        batch_size = plan.batch.b_seq_len.shape[0]
        q = q.reshape(q.shape[0], k_head_num, -1)
        if layer.params.query_scale is None:
            q, q_scale = q_per_head_fp8_quant(
                q=q, seq_lens=plan.cu_seqlens_q[1:] - plan.cu_seqlens_q[:-1], b1_start_loc=plan.cu_seqlens_q
            )
        else:
            static_q_scales = layer.params.query_scale
            q = q_per_head_static_fp8_quant(q=q, scales=static_q_scales)
            q_scale = static_q_scales.view(1, k_head_num).expand(batch_size, k_head_num)
        o = flash_attn_with_kvcache(
            q=q.reshape(-1, q_head_num, q_head_dim),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=plan.page_table,
            cache_seqlens=plan.batch.b_seq_len,
            cu_seqlens_q=plan.cu_seqlens_q,
            cu_seqlens_k_new=plan.cu_seqlens_k,
            max_seqlen_q=plan.batch.max_q_seq_len,
            causal=plan.causal,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale,
            k_descale=layer.params.key_scale.view(1, -1).expand(batch_size, -1),
            v_descale=layer.params.value_scale.view(1, -1).expand(batch_size, -1),
            return_softmax_lse=False,
        )
        return o

    def _update_decode(self, plan):
        super()._update_decode(plan)

    def _forward_decode(self, layer: LayerAttentionBinding, plan, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._decode_fp8_decode_att(
            layer, plan, q=q, k=k, v=v, alloc_func=alloc_func, logical_layer_id=layer.logical_layer_id
        )

    def _decode_fp8_decode_att(
        self,
        layer,
        plan,
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
        att_batch_size = plan.b_att_seq_len.shape[0]
        q = q.reshape(q.shape[0], k_head_num, -1)
        if layer.params.query_scale is None:
            q, q_scale = q_per_head_fp8_quant(
                q=q, seq_lens=plan.cu_seqlens_q[1:] - plan.cu_seqlens_q[:-1], b1_start_loc=plan.cu_seqlens_q
            )
        else:
            static_q_scales = layer.params.query_scale
            q = q_per_head_static_fp8_quant(q=q, scales=static_q_scales)
            q_scale = static_q_scales.view(1, k_head_num).expand(att_batch_size, k_head_num)
        o = flash_attn_with_kvcache(
            q=q.reshape(-1, q_head_num, k_head_dim),
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=plan.page_table,
            cache_seqlens=plan.b_att_seq_len,
            cu_seqlens_q=plan.cu_seqlens_q,
            cu_seqlens_k_new=plan.cu_seqlens_k,
            max_seqlen_q=plan.decode_max_q_seq_len,
            causal=plan.causal,
            window_size=(-1, -1),
            softcap=0.0,
            q_descale=q_scale.view(att_batch_size, k_head_num),
            k_descale=layer.params.key_scale.view(1, -1).expand(att_batch_size, -1),
            v_descale=layer.params.value_scale.view(1, -1).expand(att_batch_size, -1),
            return_softmax_lse=False,
        )
        return o
