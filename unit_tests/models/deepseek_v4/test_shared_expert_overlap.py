from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import deepgemm_impl
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.base_impl import FuseMoeBaseImpl
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight
from lightllm.distributed import dist_group_manager
from lightllm.models.deepseek_v4.layer_infer import transformer_layer_infer


def test_shared_expert_executes_between_combine_send_and_receive(monkeypatch):
    backend = object.__new__(FuseMoeDeepGEMM)
    backend.quant_method = SimpleNamespace(method_name="fp8w8a8-b128-deepgemm")
    backend.supports_decode_shared_expert_overlap = True
    backend.n_routed_experts = 256
    backend.expert_parallel_state = SimpleNamespace(world_size=8)
    monkeypatch.setattr(dist_group_manager, "ep_low_latency_buffer", SimpleNamespace(group_size=8))
    x = torch.ones((24, 4))
    weights = torch.ones((24, 6))
    ids = torch.zeros((24, 6), dtype=torch.int64)
    events = []
    routed = torch.full_like(x, float("nan"))

    def dispatch(*_args):
        events.append("dispatch")
        return x, None, ids, weights, None, lambda: events.append("dispatch_recv")

    def gemm(*_args, **kwargs):
        assert events[-1] == "dispatch_recv"
        assert _args[5] == 5  # ceil(24*8*6/256), based on the EP buffer's group size.
        assert kwargs["alloc_tensor_func"] is torch.empty
        events.append("gemm")
        return x * 3

    def combine(*_args):
        events.append("combine_send")

        def recv():
            events.append("combine_recv")
            routed.fill_(3)

        return routed, recv

    def shared():
        assert torch.isnan(routed).all()
        assert torch.equal(x, torch.ones_like(x))
        events.append("shared")
        return x * 2

    backend.low_latency_dispatch_with_topk = dispatch
    backend.masked_group_gemm = gemm
    backend.low_latency_combine = combine
    result = backend._fused_experts(x, None, None, weights, ids, is_prefill=False, shared_expert_func=shared)
    assert events == ["dispatch", "dispatch_recv", "gemm", "combine_send", "shared", "combine_recv"]
    assert torch.equal(result, torch.full_like(x, 5))


def test_backend_selector_only_enables_legacy_deepgemm_overlap():
    assert FuseMoeBaseImpl.supports_decode_shared_expert_overlap is False
    for mega, expected in [(None, True), ("fp8", False)]:
        backend = FuseMoeDeepGEMM(
            n_routed_experts=256,
            num_fused_shared_experts=0,
            routed_scaling_factor=1,
            quant_method=SimpleNamespace(mega_moe_mma_type=mega),
            expert_parallel_state=SimpleNamespace(eplb=None),
        )
        assert backend.supports_decode_shared_expert_overlap is expected


def test_overlap_keeps_logical_route_capture_with_weight_owner(monkeypatch):
    weight = object.__new__(fused_moe_weight.FusedMoeWeight)
    weight.layer_num_ = 0
    weight.w13 = weight.w2 = None
    weight.fuse_moe_impl = SimpleNamespace(_fused_experts=Mock(return_value="output"))
    logical_ids = torch.ones((2, 1), dtype=torch.int64)
    capture = Mock()
    monkeypatch.setattr(fused_moe_weight, "get_moe_capture_callback", lambda *_args: capture)
    shared = lambda: torch.ones((2, 4))
    weight.experts_with_topk(
        torch.ones((2, 4)),
        torch.ones((2, 1)),
        torch.zeros_like(logical_ids),
        is_prefill=False,
        logical_topk_ids=logical_ids,
        shared_expert_func=shared,
    )
    capture.assert_called_once_with(logical_ids)
    assert weight.fuse_moe_impl._fused_experts.call_args.kwargs["shared_expert_func"] is shared


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("supports_overlap", [False, True])
def test_dsv4_shared_expert_order_and_allocator(is_prefill, supports_overlap):
    layer = object.__new__(transformer_layer_infer.DeepseekV4TransformerLayerInfer)
    layer.embed_dim_ = 4
    layer.enable_ep_moe = True
    layer.moe_shared_stream = None
    layer._tpsp_allgather = lambda input, infer_state: input
    layer._select_experts = Mock(return_value=(None, None))
    layer.alloc_tensor = Mock()
    layer.swiglu_limit = 10.0
    state = SimpleNamespace(is_prefill=is_prefill)
    layer.layer_num_ = 0
    events = []

    def shared(input, *_args, **_kwargs):
        events.append("shared")
        return input * 2

    def routed(input_tensor, **kwargs):
        assert kwargs["alloc_tensor_func"] is layer.alloc_tensor
        events.append("routed")
        callback = kwargs.get("shared_expert_func")
        if callback is not None:
            events.append("combine_send")
            shared_output = callback()
            events.append("combine_recv")
            return torch.full_like(input_tensor, 3) + shared_output
        input_tensor.fill_(9)
        return torch.full_like(input_tensor, 3)

    weight = SimpleNamespace(
        experts=SimpleNamespace(
            fuse_moe_impl=SimpleNamespace(supports_decode_shared_expert_overlap=supports_overlap),
            experts_with_topk=routed,
        ),
    )
    layer._ffn_tp = shared
    out = layer._ffn(torch.ones((2, 4)), state, weight)
    expected = ["routed", "combine_send", "shared", "combine_recv"]
    assert events == (expected if supports_overlap and not is_prefill else ["shared", "routed"])
    assert torch.equal(out, torch.full_like(out, 5))


def test_dsv4_shared_expert_uses_selected_fused_activation():
    layer = object.__new__(transformer_layer_infer.DeepseekV4TransformerLayerInfer)
    layer.embed_dim_, layer.swiglu_limit = 4, 10.0
    x = torch.ones((2, 4))
    projected = torch.ones((2, 8))
    expected = torch.full_like(x, 5)
    weight = SimpleNamespace(
        gate_up_proj=SimpleNamespace(mm=Mock(return_value=projected)),
        down_proj=SimpleNamespace(mm_swiglu=Mock(return_value=expected)),
    )
    assert layer._ffn_tp(x, SimpleNamespace(), weight) is expected
    weight.down_proj.mm_swiglu.assert_called_once_with(projected, limit=10.0, use_custom_tensor_mananger=True)
