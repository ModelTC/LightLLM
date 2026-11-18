from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.triton_impl import FuseMoeTriton
from lightllm.distributed import dist_group_manager


def _overlap_backend(monkeypatch, gemm_kwargs):
    backend = object.__new__(FuseMoeDeepGEMM)
    backend.quant_method = SimpleNamespace(method_name="fp8w8a8-b128-deepgemm")
    backend.supports_decode_shared_expert_overlap = True
    backend.n_routed_experts = 256
    monkeypatch.setattr(dist_group_manager, "ep_low_latency_buffer", SimpleNamespace(group_size=8))
    x = torch.ones((4, 8))
    ids = torch.zeros((4, 2), dtype=torch.int64)
    weights = torch.ones((4, 2))
    backend.low_latency_dispatch_with_topk = lambda *_args: (x, None, ids, weights, None, lambda: None)

    def gemm(*_args, **kwargs):
        gemm_kwargs.update(kwargs)
        return x

    backend.masked_group_gemm = gemm
    backend.low_latency_combine = lambda *_args: (x * 3, lambda: None)
    return backend, x, weights, ids


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # GLM: clamped SiLU with alpha=1 and no +1 on up must reach the masked SwiGLU kernel.
        (
            dict(alpha=1.0, limit=10.0, clamp_up_add_one=False),
            dict(clamp_limit=10.0, alpha=1.0, clamp_up_add_one=False),
        ),
        # DeepSeek-V4 passes only clamp_limit; defaults must stay unchanged.
        (dict(clamp_limit=7.0), dict(clamp_limit=7.0, alpha=None, clamp_up_add_one=True)),
    ],
)
def test_overlap_forwards_activation_arguments(monkeypatch, kwargs, expected):
    gemm_kwargs = {}
    backend, x, weights, ids = _overlap_backend(monkeypatch, gemm_kwargs)
    result = backend._fused_experts(
        x, None, None, weights, ids, is_prefill=False, shared_expert_func=lambda: x * 2, **kwargs
    )
    for key, value in expected.items():
        assert gemm_kwargs[key] == value, key
    assert torch.equal(result, x * 5)


def test_base_call_only_forwards_shared_expert_func_when_given():
    backend = object.__new__(FuseMoeTriton)
    backend._select_experts = Mock(return_value=("w", "ids", "ids"))
    # Backends without overlap support do not accept shared_expert_func.
    backend._fused_experts = Mock(side_effect=lambda **kwargs: kwargs)
    common = dict(
        input_tensor=None,
        router_logits=None,
        w13=None,
        w2=None,
        correction_bias=None,
        scoring_func="sigmoid",
        top_k=2,
        renormalize=True,
        use_grouped_topk=False,
        topk_group=1,
        num_expert_group=1,
        is_prefill=False,
    )
    assert "shared_expert_func" not in backend(**common)
    shared = lambda: None
    assert backend(**common, shared_expert_func=shared)["shared_expert_func"] is shared
