from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends._flashinfer import set_flashinfer_envs
from lightllm.common.attention.backends.dense.flashinfer.backend import (
    FlashInferDensePrefillBackend,
    FlashInferDenseDecodeBackend,
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


class FlashInferDenseFp8Common:
    family = AttentionFamily.DENSE
    supports_expert_parallel = False

    provider = "flashinfer"

    cache_encodings = frozenset({CacheEncoding.FP8_TENSOR})

    def __init__(self, env, spec):
        set_flashinfer_envs()
        super().__init__(env=env, spec=spec)
        self.kv_data_type = torch.float8_e4m3fn

    @property
    def kernel_page_size(self) -> int:
        return 1


class FlashInferDenseFp8PrefillBackend(FlashInferDenseFp8Common, FlashInferDensePrefillBackend):
    phase = AttentionPhase.PREFILL
    supported_features = frozenset(
        {AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.NONCAUSAL, AttentionFeature.MTP_WINDOW}
    )

    def _run_prefill(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype, device=q.device)
        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        k_descale = layer.resources.key_scale
        v_descale = layer.resources.value_scale
        context.prefill_wrapper.run(q=q, paged_kv_cache=(k, v), k_scale=k_descale, v_scale=v_descale, out=o_tensor)
        return o_tensor

    def allocate_context(self, capacity, *, capture, workspace) -> Fp8FlashInferPrefillContext:
        return Fp8FlashInferPrefillContext(capacity, capture, workspace)

    def forward(
        self, layer: LayerAttention, context: Fp8FlashInferPrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._run_prefill(layer, context, q=q, k=k, v=v)


class FlashInferDenseFp8DecodeBackend(FlashInferDenseFp8Common, FlashInferDenseDecodeBackend):
    phase = AttentionPhase.DECODE
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    def _run_decode(
        self,
        layer,
        context,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        o_tensor = self.alloc_tensor(q.shape, q.dtype, device=q.device)
        k = k.unsqueeze(1).view(torch.float8_e4m3fn)
        v = v.unsqueeze(1).view(torch.float8_e4m3fn)
        k_descale = layer.resources.key_scale
        v_descale = layer.resources.value_scale
        context.decode_wrapper.run(q=q, paged_kv_cache=(k, v), k_scale=k_descale, v_scale=v_descale, out=o_tensor)
        return o_tensor

    def allocate_context(self, capacity, *, capture, workspace) -> Fp8FlashInferDecodeContext:
        return Fp8FlashInferDecodeContext(capacity, capture, workspace)

    def forward(
        self, layer: LayerAttention, context: Fp8FlashInferDecodeContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        return self._run_decode(layer, context, q=q, k=k, v=v)
