"""Build FP8 GQA adapter states whose only replaced parts are the external FA3/FlashInfer kernels."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeStepArgs, FA3GQAPrefillStepArgs
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferGQABuilderConfig,
    FlashInferGQADecodeStepArgs,
    FlashInferGQAPrefillStepArgs,
)


def fp8_attention_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str,
    prefill: bool,
    infer,
    observed: dict,
    heads: int,
    head_dim: int,
    requests: int,
):
    """Return an adapter attention state for kind 'static_head' (FA3) or 'static_tensor' (FlashInfer).

    The kernel records its keyword arguments in observed (plus 'q_scales' for the static q quantizer)
    and copies q to out. requests is the number of attention requests the step args describe.
    """
    import lightllm.utils.sgl_utils as sgl_utils
    from lightllm.common.basemodel.attention.fa3 import fp as fa3_fp
    from lightllm.common.basemodel.attention.fa3.fp8 import Fp8Fa3AttBackend
    from lightllm.common.basemodel.attention.flashinfer import fp as fi_fp
    from lightllm.common.basemodel.attention.flashinfer.fp8 import Fp8FlashInferAttBackend
    from lightllm.common.basemodel.triton_kernel.quantization import q_per_head_fp8_quant

    def ref():
        return infer

    if kind == "static_head":

        def quantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            observed["q_scales"] = scales
            return q

        def attend(**kwargs):
            observed.update(kwargs)
            kwargs["out"].copy_(kwargs["q"])

        monkeypatch.setattr(q_per_head_fp8_quant, "q_per_head_static_fp8_quant", quantize)
        monkeypatch.setattr(sgl_utils, "flash_attn_with_kvcache", attend)
        state_cls = fa3_fp.Fa3PrefillAttState if prefill else fa3_fp.Fa3DecodeAttState
        state = state_cls(backend=None, _infer_state_weak_ref=ref)
        step_args_cls, extra = (FA3GQAPrefillStepArgs, {}) if prefill else (FA3GQADecodeStepArgs, {"max_seqlen_k": 1})
        state.step_args = step_args_cls(
            page_table=None,
            cache_seqlens=torch.ones(requests, dtype=torch.int32),
            cu_seqlens_q=torch.arange(requests + 1, dtype=torch.int32),
            cu_seqlens_k=None,
            max_seqlen_q=1,
            page_size=1,
            **extra,
        )
        kv_quant = Fp8Fa3AttBackend.kv_quant
    else:

        def run(q, paged_kv_cache, k_scale, v_scale, out):
            observed.update(k_scale=k_scale, v_scale=v_scale)
            out.copy_(q)

        state_cls = fi_fp.FlashInferPrefillAttState if prefill else fi_fp.FlashInferDecodeAttState
        state = state_cls(backend=None, _infer_state_weak_ref=ref)
        config = FlashInferGQABuilderConfig(heads, heads, head_dim, 1, torch.bfloat16, torch.float8_e4m3fn)
        step_args_cls = FlashInferGQAPrefillStepArgs if prefill else FlashInferGQADecodeStepArgs
        state.step_args = step_args_cls(wrapper=SimpleNamespace(run=run), config=config)
        kv_quant = Fp8FlashInferAttBackend.kv_quant
    state.causal = True
    state.backend = SimpleNamespace(
        kv_quant=kv_quant, get_attention_impl=lambda backend_cls, layer_config: backend_cls.get_impl_cls()(layer_config)
    )
    return state
