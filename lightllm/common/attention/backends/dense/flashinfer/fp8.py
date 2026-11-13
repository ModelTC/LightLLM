import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends._flashinfer import set_flashinfer_envs
from lightllm.common.attention.backends.dense.flashinfer.backend import (
    FlashInferAttBackend,
    FlashInferDecodeContext,
    FlashInferPrefillContext,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(slots=True)
class Fp8FlashInferPrefillContext(FlashInferPrefillContext):
    pass


@dataclasses.dataclass(slots=True)
class Fp8FlashInferDecodeContext(FlashInferDecodeContext):
    pass


class Fp8FlashInferAttBackend(FlashInferAttBackend):
    provider = "flashinfer"
    cache_formats = frozenset({"fp8kv_spt"})

    def __init__(self, model, spec):
        set_flashinfer_envs()
        super().__init__(model=model, spec=spec)
        self.kv_data_type = torch.float8_e4m3fn

    @property
    def infer_page_size(self) -> int:
        return 1

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return Fp8FlashInferPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return Fp8FlashInferDecodeContext(capacity, capture, workspace)

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._run_prefill(
            layer, context, q=q, k=k, v=v, alloc_func=alloc_func, logical_layer_id=layer.logical_layer_id
        )

    def _run_prefill(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logical_layer_id: int,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        o_tensor = alloc_func(q.shape, q.dtype, device=q.device)
        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        k_descale = layer.params.key_scale
        v_descale = layer.params.value_scale
        context.prefill_wrapper.run(q=q, paged_kv_cache=(k, v), k_scale=k_descale, v_scale=v_descale, out=o_tensor)
        return o_tensor

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None):
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        return self._run_decode(
            layer, context, q=q, k=k, v=v, alloc_func=alloc_func, logical_layer_id=layer.logical_layer_id
        )

    def _run_decode(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logical_layer_id: int,
        alloc_func=torch.empty,
    ):
        o_tensor = alloc_func(q.shape, q.dtype, device=q.device)
        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        k_descale = layer.params.key_scale
        v_descale = layer.params.value_scale
        context.decode_wrapper.run(q=q, paged_kv_cache=(k, v), k_scale=k_descale, v_scale=v_descale, out=o_tensor)
        return o_tensor
