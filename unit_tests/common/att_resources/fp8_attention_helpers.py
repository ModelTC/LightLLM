"""Run FP8 GQA attention whose only replaced parts are the external FA3/FlashInfer kernels."""

from unit_tests.common.attention.backends.groups_runner import run_attention

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.gqa import GQAAttConfig, GQAAttInputs
from lightllm.common.attention.backends.gqa.fa3 import (
    FA3GQADecodeBackend,
    FA3GQADecodeAttState,
    FA3GQAPrefillBackend,
    FA3GQAPrefillAttState,
)
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferGQADecodeBackend,
    FlashInferGQADecodeAttState,
    FlashInferGQAPrefillBackend,
    FlashInferGQAPrefillAttState,
)
from lightllm.common.attention.backends.gqa import get_gqa_kv_quant


class FP8Attention:
    """One phase of FP8 GQA attention over given attention state; call run(q, k, v, kv_scales, alloc_func)."""

    def __init__(self, backend, kv_type: str, att_state) -> None:
        self.backend, self.kv_type, self.att_state = backend, kv_type, att_state

    def run(self, *, q, k, v, kv_scales, alloc_func=torch.empty):
        # 该层从其缓存中创建的层配置在模型加载时会根据缓存比例进行调整。
        kv_quant, group_size, has_q_scale = get_gqa_kv_quant(
            llm_kv_type=self.kv_type, kv_scales=kv_scales, head_dim=q.shape[2]
        )
        att_config = GQAAttConfig(
            num_heads=q.shape[1],
            num_kv_heads=k.shape[1],
            head_dim=q.shape[2],
            dtype=q.dtype,
            kv_quant=kv_quant,
            kv_quant_group_size=group_size,
            has_q_scale=has_q_scale,
        )
        if self.backend in (FlashInferGQAPrefillBackend, FlashInferGQADecodeBackend):
            runtime = self.backend.RuntimeConfig(max_seq_len=k.shape[0])
            impl = self.backend(att_config=att_config, runtime_config=runtime)
        else:
            impl = self.backend(att_config=att_config)
        inputs = GQAAttInputs(q=q, k_cache=k, v_cache=v, kv_scales=kv_scales)
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return run_attention(
            backend=impl, att_state=self.att_state, layer_inputs=inputs, out=out, alloc_func=alloc_func
        )


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
        cache_seqlens = torch.ones(size=(requests,), dtype=torch.int32)
        cu_q = torch.arange(end=requests + 1, dtype=torch.int32)
        if prefill:
            att_state = FA3GQAPrefillAttState(
                page_table=None, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_q, cu_seqlens_k=None, max_seqlen_q=1
            )
        else:
            att_state = FA3GQADecodeAttState(
                page_table=None,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=None,
                max_seqlen_q=1,
                max_seqlen_k=1,
            )
        return FP8Attention(FA3GQAPrefillBackend if prefill else FA3GQADecodeBackend, "fp8kv_sph", att_state)
    else:

        def run(q, paged_kv_cache, k_scale, v_scale, out):
            observed.update(k_scale=k_scale, v_scale=v_scale)
            out.copy_(q)

        wrapper = SimpleNamespace(run=run)
        if prefill:
            att_state = FlashInferGQAPrefillAttState(wrapper=wrapper)
        else:
            att_state = FlashInferGQADecodeAttState(
                wrapper=wrapper,
                kv_starts=torch.arange(end=requests + 1, dtype=torch.int32),
                kv_indices=torch.arange(end=requests, dtype=torch.int32),
                kv_last_page_len=torch.ones(size=(requests,), dtype=torch.int32),
            )
        backend = FlashInferGQAPrefillBackend if prefill else FlashInferGQADecodeBackend
        return FP8Attention(backend, "fp8kv_spt", att_state)
