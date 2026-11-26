"""Run FP8 GQA attention whose only replaced parts are the external FA3/FlashInfer kernels."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.fa3 import (
    FA3GQADecodeBackend,
    FA3GQADecodeAttState,
    FA3GQAPrefillBackend,
    FA3GQAPrefillAttState,
)
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferGQABuilderConfig,
    FlashInferGQADecodeBackend,
    FlashInferGQADecodeAttState,
    FlashInferGQAPrefillBackend,
    FlashInferGQAPrefillAttState,
)
from lightllm.common.attention.backends.gqa import gqa_kv_quant_fields


class FP8Attention:
    """One phase of FP8 GQA attention over given attention state; call run(q, k, v, kv_scales, alloc_func)."""

    def __init__(self, backend, kv_type: str, att_state) -> None:
        self.backend, self.kv_type, self.att_state = backend, kv_type, att_state

    def run(self, *, q, k, v, kv_scales, alloc_func=torch.empty):
        # The layer config a layer creates from its cache scales when the model loads.
        layer_config = GQALayerConfig(
            num_heads=q.shape[1],
            num_kv_heads=k.shape[1],
            head_dim=q.shape[2],
            dtype=q.dtype,
            **gqa_kv_quant_fields(self.kv_type, kv_scales, q.shape[2]),
        )
        backend = self.backend(layer_config)
        inputs = GQALayerInputs(q=q, k_cache=k, v_cache=v, kv_scales=kv_scales)
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return backend.forward(self.att_state, inputs, out=out, alloc_func=alloc_func)


def fp8_attention(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str,
    prefill: bool,
    observed: dict,
    heads: int,
    head_dim: int,
    requests: int,
):
    """Return FP8Attention for kind 'static_head' (FA3) or 'static_tensor' (FlashInfer).

    The kernel records its keyword arguments in observed (plus 'q_scales' for the static q quantizer)
    and copies q to out. requests is the number of attention requests the attention state describe.
    """
    import lightllm.utils.sgl_utils as sgl_utils
    from lightllm.common.basemodel.triton_kernel.quantization import q_per_head_fp8_quant

    if kind == "static_head":

        def quantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            observed["q_scales"] = scales
            return q

        def attend(**kwargs):
            observed.update(kwargs)
            kwargs["out"].copy_(kwargs["q"])

        monkeypatch.setattr(q_per_head_fp8_quant, "q_per_head_static_fp8_quant", quantize)
        monkeypatch.setattr(sgl_utils, "flash_attn_with_kvcache", attend)
        att_state_cls, extra = (FA3GQAPrefillAttState, {}) if prefill else (FA3GQADecodeAttState, {"max_seqlen_k": 1})
        att_state = att_state_cls(
            page_table=None,
            cache_seqlens=torch.ones(requests, dtype=torch.int32),
            cu_seqlens_q=torch.arange(requests + 1, dtype=torch.int32),
            cu_seqlens_k=None,
            max_seqlen_q=1,
            page_size=1,
            **extra,
        )
        return FP8Attention(FA3GQAPrefillBackend if prefill else FA3GQADecodeBackend, "fp8kv_sph", att_state)
    else:

        def run(q, paged_kv_cache, k_scale, v_scale, out):
            observed.update(k_scale=k_scale, v_scale=v_scale)
            out.copy_(q)

        config = FlashInferGQABuilderConfig(heads, heads, head_dim, 1, torch.bfloat16, torch.float8_e4m3fn)
        att_state_cls = FlashInferGQAPrefillAttState if prefill else FlashInferGQADecodeAttState
        att_state = att_state_cls(wrapper=SimpleNamespace(run=run), config=config)
        backend = FlashInferGQAPrefillBackend if prefill else FlashInferGQADecodeBackend
        return FP8Attention(backend, "fp8kv_spt", att_state)
