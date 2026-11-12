import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends._flashinfer import set_flashinfer_envs
from lightllm.common.attention.backends.dense.flashinfer.backend import (
    FlashInferAttBackend,
    FlashInferDecodePlan,
    FlashInferPrefillPlan,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class Fp8FlashInferPrefillPlan(FlashInferPrefillPlan):
    pass


@dataclasses.dataclass
class Fp8FlashInferDecodePlan(FlashInferDecodePlan):
    pass


class Fp8FlashInferAttBackend(FlashInferAttBackend):
    provider = "flashinfer"
    cache_formats = frozenset({"fp8kv_spt"})

    def __init__(self, model):
        set_flashinfer_envs()
        super().__init__(model=model)
        self.kv_data_type = torch.float8_e4m3fn

    def _init_infer_page_size(self):
        self.infer_page_size = 1
        assert (
            self.page_size % self.infer_page_size == 0
        ), f"model page_size {self.page_size} must be divisible by infer_page_size {self.infer_page_size}"
        logger.warning(
            f"Fp8FlashInferAttBackend temporarily uses infer_page_size=1 with model page_size={self.page_size}; "
            "multi-token FP8 inference pages are not implemented yet."
        )

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return Fp8FlashInferPrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return Fp8FlashInferDecodePlan(capacity, capture, workspace)

    def _forward_prefill(self, layer: LayerAttention, plan, inputs: DenseInputs, *, out=None) -> torch.Tensor:
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
        o_tensor = alloc_func(q.shape, q.dtype, device="cuda")
        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        k_descale = layer.params.key_scale
        v_descale = layer.params.value_scale
        plan.prefill_wrapper.run(q=q, paged_kv_cache=(k, v), k_scale=k_descale, v_scale=v_descale, out=o_tensor)
        return o_tensor

    def _forward_decode(self, layer: LayerAttention, plan, inputs: DenseInputs, *, out=None):
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
        o_tensor = alloc_func(q.shape, q.dtype, device="cuda")
        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        k_descale = layer.params.key_scale
        v_descale = layer.params.value_scale
        plan.decode_wrapper.run(q=q, paged_kv_cache=(k, v), k_scale=k_descale, v_scale=v_descale, out=o_tensor)
        return o_tensor
