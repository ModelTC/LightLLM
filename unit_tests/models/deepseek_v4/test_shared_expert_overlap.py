"""Shared-expert callback, dispatch and model fallback contracts."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import base_impl, deepgemm_impl
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.triton_impl import FuseMoeTriton
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight as weight_module
from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer


@pytest.mark.parametrize("overlap", [False, True])
def test_real_decode_pipeline_waits_after_shared_computation(monkeypatch, overlap):
    events = []
    x = torch.ones(3, 4, dtype=torch.bfloat16)
    routed = torch.full_like(x, float("nan"))

    class Buffer:
        group_size = 8

        def low_latency_dispatch(self, *args, **kwargs):
            assert kwargs["return_recv_hook"] is False
            events.append("dispatch")
            return x, None, "handle", None, None

        def low_latency_combine(self, *args, **kwargs):
            events.append("combine_send")
            assert kwargs["return_recv_hook"] is overlap

            def receive():
                events.append("combine_recv")
                routed.fill_(3)

            if not overlap:
                receive()
            return routed, None, receive

    def gemm(*args, **kwargs):
        assert args[7] == 3  # ceil(3 requests * 8 ranks * topk2 / 16 experts)
        events.append("gemm")
        return x

    def shared():
        assert torch.isnan(routed).all()
        assert torch.all(x == 1)
        events.append("shared")
        return x * 2

    monkeypatch.setattr(ep, "masked_group_gemm", gemm)
    monkeypatch.setattr(ep, "get_env_use_ue8m0_scales", lambda: False)
    monkeypatch.setattr(ep, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 64)
    out = ep.fused_experts_impl(
        x,
        torch.ones(2, 8, 4),
        torch.ones(2, 4, 4),
        torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int64),
        16,
        Buffer(),
        False,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        shared_expert_func=shared if overlap else None,
    )
    expected = ["dispatch", "gemm", "combine_send"] + (["shared", "combine_recv"] if overlap else ["combine_recv"])
    assert events == expected
    assert torch.all(out == (5 if overlap else 3))


@pytest.mark.parametrize("mega", [False, True])
def test_capability_is_selected_at_backend_construction(monkeypatch, mega):
    monkeypatch.setattr(base_impl, "get_global_rank", lambda: 0)
    monkeypatch.setattr(base_impl, "get_global_world_size", lambda: 8)
    monkeypatch.setattr(deepgemm_impl, "use_sm100_mega_moe", lambda _: mega)
    backend = deepgemm_impl.FuseMoeDeepGEMM(16, 0, 1.0, SimpleNamespace(), 0, None, None, False)
    assert backend.supports_decode_shared_expert_overlap is (not mega)
    assert base_impl.FuseMoeBaseImpl.supports_decode_shared_expert_overlap is False


def test_weight_preserves_route_capture_and_callback(monkeypatch):
    owner = FusedMoeWeight.__new__(FusedMoeWeight)
    owner.layer_num_ = 0
    owner.w13 = owner.w2 = None
    backend = deepgemm_impl.FuseMoeDeepGEMM.__new__(deepgemm_impl.FuseMoeDeepGEMM)
    backend.supports_decode_shared_expert_overlap = True
    backend._fused_experts = Mock(return_value="output")
    owner.fuse_moe_impl = backend
    capture = Mock()
    monkeypatch.setattr(weight_module, "get_moe_capture_callback", lambda *_: capture)
    ids = torch.zeros(3, 2, dtype=torch.int64)
    shared = Mock()
    assert (
        owner.experts_with_topk(torch.ones(3, 4), torch.ones(3, 2), ids, is_prefill=False, shared_expert_func=shared)
        == "output"
    )
    capture.assert_called_once_with(ids)
    assert backend._fused_experts.call_args.kwargs["shared_expert_func"] is shared


def test_existing_inplace_backends_receive_no_new_keyword():
    backend = FuseMoeTriton.__new__(FuseMoeTriton)
    backend._fused_experts = Mock(return_value="old-output")
    assert backend.fused_experts_with_topk(None, None, None, None, None) == "old-output"
    assert "shared_expert_func" not in backend._fused_experts.call_args.kwargs
    with pytest.raises(AssertionError):
        backend.fused_experts_with_topk(None, None, None, None, None, shared_expert_func=lambda: None)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ep_mode", [False, True])
@pytest.mark.parametrize("prefill,supports", [(False, True), (False, False), (True, True)])
def test_model_keeps_shared_before_inplace_fallback(prefill, supports, enabled, ep_mode):
    layer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    layer.embed_dim_ = 4
    layer.enable_ep_moe = ep_mode
    layer.swiglu_limit = 7
    layer.enable_shared_expert_overlap = enabled
    layer._tpsp_allgather = lambda input, infer_state: input
    ids = torch.zeros(3, 2, dtype=torch.int64)
    layer._select_experts = lambda *args: (torch.ones(3, 2), ids)
    events = []

    def shared(x=None, *args, **kwargs):
        if x is None:
            x = kwargs["input"]
        events.append("shared")
        return x * 2

    layer._ffn_tp = shared

    def routed(**kwargs):
        events.append("routed")
        if kwargs.get("shared_expert_func") is not None:
            return kwargs["shared_expert_func"]() + 3
        kwargs["input_tensor"].fill_(9)
        return torch.full((3, 4), 3.0)

    owner = FusedMoeWeight.__new__(FusedMoeWeight)
    owner.fuse_moe_impl = SimpleNamespace(supports_decode_shared_expert_overlap=supports)
    owner.experts_with_topk = routed
    weights = SimpleNamespace(experts=owner)
    reductions = []
    layer._tpsp_reduce = lambda input, infer_state: (reductions.append(input) or input)
    result = layer._ffn(torch.ones(3, 4), SimpleNamespace(is_prefill=prefill), weights)
    assert torch.all(result == 5)
    assert events == (["routed", "shared"] if enabled and supports and not prefill else ["shared", "routed"])
    assert len(reductions) == (0 if ep_mode else 1)


@pytest.mark.parametrize("supported", [False, True])
def test_weight_reports_backend_capability(supported):
    owner = FusedMoeWeight.__new__(FusedMoeWeight)
    owner.fuse_moe_impl = SimpleNamespace(supports_decode_shared_expert_overlap=supported)
    assert owner.supports_decode_shared_expert_overlap is supported


@pytest.mark.parametrize("value,enabled", [(None, True), ("0", False), ("1", True), ("TRUE", True)])
def test_overlap_switch_is_read_when_layer_is_created(monkeypatch, value, enabled):
    from lightllm.models.deepseek_v4.layer_infer import transformer_layer_infer

    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    if value is None:
        monkeypatch.delenv("LIGHTLLM_DSV4_SHARED_EXPERT_OVERLAP", raising=False)
    else:
        monkeypatch.setenv("LIGHTLLM_DSV4_SHARED_EXPERT_OVERLAP", value)
    monkeypatch.setattr(transformer_layer_infer, "get_env_start_args", lambda: SimpleNamespace(enable_ep_moe=True))
    config = dict(
        rms_norm_eps=1e-6,
        hidden_size=4096,
        num_attention_heads=32,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        compress_ratios=[0],
        num_hash_layers=0,
        vocab_size=129280,
        n_layer=1,
        num_experts_per_tok=6,
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
    )
    layer = DeepseekV4TransformerLayerInfer(0, config)
    assert layer.enable_shared_expert_overlap is enabled
