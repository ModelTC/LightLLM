from types import SimpleNamespace

import pytest
import torch

from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight as weight_module
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import deepgemm_impl
from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("shared_mode", ["absent", "separate", "native_fused"])
def test_shared_output_is_added_once(prefill, enabled, shared_mode):
    hidden = torch.zeros(2, 128, dtype=torch.bfloat16)
    shared = torch.full_like(hidden, 0.25)
    calls = []

    def experts(input_tensor, **kwargs):
        calls.append(kwargs)
        routed = torch.ones_like(input_tensor)
        return routed if "shared_expert_output" not in kwargs else routed + kwargs["shared_expert_output"]

    layer = SimpleNamespace(
        embed_dim_=128,
        n_shared_experts=None if shared_mode == "absent" else 1,
        ep_shared_combine_bias=enabled,
        swiglu_limit=7.0,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        _ffn_tp=lambda *args: shared,
    )
    weights = SimpleNamespace(
        num_fused_shared_experts=int(shared_mode == "native_fused"),
        moe_gate=SimpleNamespace(data_type_=torch.float32, mm=lambda x: torch.zeros(2, 4)),
        experts=SimpleNamespace(experts=experts),
    )
    result = Glm5NextTransformerLayerInfer._moe_ffn(layer, hidden, SimpleNamespace(is_prefill=prefill), weights)
    expected = torch.ones_like(hidden) + (shared if shared_mode == "separate" else 0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if enabled and prefill and shared_mode == "separate":
        assert calls[0]["shared_expert_output"] is shared
    else:
        assert "shared_expert_output" not in calls[0]


@pytest.mark.parametrize("with_shared", [False, True])
def test_shared_bias_reaches_normal_combine_backend(monkeypatch, with_shared):
    hidden = torch.zeros(2, 128, dtype=torch.bfloat16)
    shared = torch.ones_like(hidden) if with_shared else None
    ids = torch.zeros(2, 1, dtype=torch.int32)
    impl = object.__new__(deepgemm_impl.FuseMoeDeepGEMM)
    impl._select_experts = lambda **kwargs: (torch.ones(2, 1), ids, ids)
    impl.total_expert_num_contain_redundancy = 8
    impl.quant_method = SimpleNamespace(block_size=128)
    pack = SimpleNamespace(weight=hidden, weight_scale=None)
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return hidden

    monkeypatch.setattr(weight_module, "get_moe_capture_callback", lambda *args: None)
    monkeypatch.setattr(ep, "check_ep_expert_dtype", lambda method: None)
    monkeypatch.setattr(ep, "use_sm100_mega_moe", lambda method: False)
    monkeypatch.setattr(ep, "fused_experts_impl", capture)
    monkeypatch.setattr(ep.dist_group_manager, "ep_buffer", object(), raising=False)
    weight = SimpleNamespace(
        layer_num_=3,
        fuse_moe_impl=impl,
        w13=pack,
        w2=pack,
        e_score_correction_bias=None,
        scoring_func="sigmoid",
        per_expert_scale=None,
    )
    result = weight_module.FusedMoeWeight.experts(
        weight,
        hidden,
        hidden,
        top_k=1,
        renormalize=True,
        use_grouped_topk=False,
        topk_group=1,
        num_expert_group=1,
        is_prefill=True,
        shared_expert_output=shared,
    )
    assert result is hidden and calls[0]["shared_expert_output"] is shared


def test_unsupported_shared_bias_fails_before_dispatch(monkeypatch):
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    args = dict(
        hidden_states=x,
        w13=None,
        w2=None,
        topk_weights=None,
        topk_idx=None,
        num_experts=8,
        quant_method=None,
        shared_expert_output=x,
    )
    with pytest.raises(ValueError, match="normal prefill combine"):
        ep.fused_experts(**args, is_prefill=False)
    monkeypatch.setattr(ep, "check_ep_expert_dtype", lambda method: None)
    monkeypatch.setattr(ep, "use_sm100_mega_moe", lambda method: True)
    with pytest.raises(NotImplementedError, match="DeepEP normal combine"):
        ep.fused_experts(**args, is_prefill=True)


@pytest.mark.parametrize("with_shared", [False, True])
def test_normal_combine_preserves_default_call_contract(monkeypatch, with_shared):
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    bias = torch.ones_like(x) if with_shared else None
    calls = []
    handle = SimpleNamespace(num_recv_tokens_per_expert_list=[0], recv_src_metadata=torch.empty(0, 3))

    def combine(*args, **kwargs):
        calls.append(kwargs)
        return x, None, None

    buffer = SimpleNamespace(dispatch=lambda *args, **kwargs: ((x, x), None, None, handle, None), combine=combine)
    monkeypatch.setattr(ep, "per_token_group_quant_fp8", lambda *args, **kwargs: (x, x))
    monkeypatch.setattr(ep, "get_deepep_num_max_dispatch_tokens_per_rank_prefill", lambda: 256)
    monkeypatch.setattr(ep.Autotuner, "is_kernel_autotune_warmup", lambda *args: False)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    result = ep.fused_experts_impl(
        x,
        torch.empty(1, 256, 128),
        torch.empty(1, 128, 128),
        torch.ones(2, 1),
        ids,
        num_experts=1,
        buffer=buffer,
        is_prefill=True,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        shared_expert_output=bias,
    )
    assert result is x
    if with_shared:
        assert calls[0]["bias"] is bias
    else:
        assert "bias" not in calls[0]
