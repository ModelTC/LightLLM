"""Shared-expert overlap on the EP prefill and decode paths."""

from types import SimpleNamespace
import pytest
import torch
import lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep as ep_module
from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import fused_experts_impl
from lightllm.models.deepseek_v4.layer_infer import transformer_layer_infer as layer_module
from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight


def _layer(ep_mode):
    layer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    layer.embed_dim_ = 4
    layer.layer_num_ = 0
    layer.enable_ep_moe = ep_mode
    layer.swiglu_limit = 7
    layer._tpsp_allgather = lambda input, infer_state: input
    ids = torch.zeros(3, 2, dtype=torch.int64)
    layer._select_experts = lambda *args: (torch.ones(3, 2), ids)
    return layer, ids


def _shared_recorder(events, scale=2):
    def shared(*args, **kwargs):
        x = kwargs["input"] if "input" in kwargs else args[0]
        events.append("shared")
        return x * scale

    return shared


def _moe_inputs():
    hidden = torch.ones(3, 4)
    topk_weights = torch.ones(3, 2)
    topk_idx = torch.zeros(3, 2, dtype=torch.int64)
    w1 = torch.ones(2, 8, 4)
    w2 = torch.ones(2, 4, 4)
    return hidden, topk_weights, topk_idx, w1, w2


def test_ep_ffn_passes_the_shared_expert_callback(monkeypatch):
    monkeypatch.setattr(layer_module, "use_sm100_mega_moe", lambda method: False)
    layer, _ids = _layer(True)
    events = []
    layer._ffn_shared_expert = _shared_recorder(events)

    def routed(**kwargs):
        events.append("routed")
        return torch.full((3, 4), 3.0) + kwargs["shared_expert_func"]()

    experts = SimpleNamespace(quant_method=None, experts_with_topk=routed)
    result = layer._ffn(
        torch.ones(3, 4),
        SimpleNamespace(is_prefill=False),
        SimpleNamespace(experts=experts),
    )
    assert events == ["routed", "shared"]
    assert torch.all(result == 5)


def test_decode_runs_shared_expert_between_combine_send_and_recv(monkeypatch):
    events = []
    hidden, topk_weights, topk_idx, w1, w2 = _moe_inputs()

    def dispatch(*args, **kwargs):
        events.append("dispatch")
        return (hidden, hidden), torch.ones(2, dtype=torch.int32), "handle", None, None

    def gemm(*args, **kwargs):
        assert args[7] == 3  # ceil(3 tokens * group 8 * topk 2 / 16 experts)
        events.append("gemm")
        return hidden

    def combine(*args, **kwargs):
        assert kwargs["return_recv_hook"] is True
        events.append("combine_send")

        def receive():
            events.append("combine_recv")

        return torch.full((3, 4), 3.0), None, receive

    monkeypatch.setattr(ep_module, "masked_group_gemm", gemm)
    result = fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_idx,
        num_experts=16,
        buffer=SimpleNamespace(group_size=8, low_latency_dispatch=dispatch, low_latency_combine=combine),
        is_prefill=False,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        shared_expert_func=lambda: events.append("shared") or torch.full((3, 4), 2.0),
    )
    assert events == ["dispatch", "gemm", "combine_send", "shared", "combine_recv"]
    assert torch.all(result == 5)


def test_prefill_overlaps_shared_expert_with_combine(monkeypatch):
    events = []
    hidden, topk_weights, topk_idx, w1, w2 = _moe_inputs()
    import deep_ep

    monkeypatch.setattr(deep_ep.ElasticBuffer, "capture", staticmethod(lambda: events.append("event") or "event"))
    monkeypatch.setattr(ep_module, "get_deepep_num_max_dispatch_tokens_per_rank_prefill", lambda: 16)
    monkeypatch.setattr(ep_module, "per_token_group_quant_fp8", lambda *args, **kwargs: (hidden, hidden))

    def dispatch(*args, **kwargs):
        events.append("dispatch")
        handle = SimpleNamespace(
            num_recv_tokens_per_expert_list=[0],
            recv_src_metadata=torch.zeros(3, 4),
        )
        return (hidden, hidden), None, topk_weights, handle, None

    class _Event:
        def current_stream_wait(self):
            events.append("combine_recv")

    def combine(*args, **kwargs):
        assert kwargs["previous_event"] == "event"
        assert kwargs["async_with_compute_stream"] is True
        events.append("combine_send")
        return torch.full((3, 4), 3.0), None, _Event()

    result = fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_idx,
        num_experts=16,
        buffer=SimpleNamespace(dispatch=dispatch, combine=combine),
        is_prefill=True,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        shared_expert_func=lambda: events.append("shared") or torch.full((3, 4), 2.0),
    )
    assert events == ["dispatch", "event", "combine_send", "shared", "combine_recv"]
    assert torch.all(result == 5)


def test_rename_shared_expert_as_the_last_routed_expert():
    weight = DeepseekV4TransformerLayerWeight.__new__(DeepseekV4TransformerLayerWeight)
    weight.prefix = "layers.0"
    weight.n_routed_experts = 4
    weight.experts = SimpleNamespace(
        quant_method=SimpleNamespace(weight_scale_suffix="weight_scale_inv", weight_zero_point_suffix=None)
    )
    tensors = {
        "layers.0.ffn.shared_experts.w1.weight": torch.ones(2),
        "layers.0.ffn.shared_experts.w1.scale": torch.ones(1),
        "layers.0.ffn.shared_experts.w2.weight": torch.ones(2),
        "layers.0.ffn.shared_experts.w3.weight": torch.ones(2),
    }
    weight._rename_shared_experts(tensors)
    assert torch.equal(tensors["layers.0.ffn.experts.4.w1.weight"], torch.ones(2))
    assert torch.equal(tensors["layers.0.ffn.experts.4.w1.scale"], torch.ones(1))
    assert "layers.0.ffn.experts.4.w2.weight" in tensors
    assert "layers.0.ffn.experts.4.w3.weight" in tensors


def test_auto_enable_fused_shared_experts_for_v4_tp(monkeypatch):
    from lightllm.utils import config_utils

    monkeypatch.setattr(config_utils, "get_model_type", lambda path: "deepseek_v4")
    args = SimpleNamespace(model_dir="/model", enable_ep_moe=False, enable_fused_shared_experts=False)
    config_utils.auto_set_fused_shared_experts(args)
    assert args.enable_fused_shared_experts is True

    args.enable_ep_moe = True
    args.enable_fused_shared_experts = False
    config_utils.auto_set_fused_shared_experts(args)
    assert args.enable_fused_shared_experts is False


def test_tp_fused_shared_expert_skips_the_standalone_ffn(monkeypatch):
    monkeypatch.setattr(layer_module, "use_sm100_mega_moe", lambda method: False)
    layer, _ids = _layer(False)
    events = []
    layer._ffn_shared_expert = _shared_recorder(events)
    reductions = []
    layer._tpsp_reduce = lambda input, infer_state: reductions.append(input.clone()) or input

    def routed(**kwargs):
        events.append("routed")
        return torch.full((3, 4), 3.0)

    experts = SimpleNamespace(quant_method=None, experts_with_topk=routed)
    result = layer._ffn(
        torch.ones(3, 4),
        SimpleNamespace(is_prefill=False),
        SimpleNamespace(experts=experts, num_fused_shared_experts=1),
    )
    assert events == ["routed"]
    assert torch.all(result == 3)
    assert len(reductions) == 1
    assert torch.all(reductions[0] == 3)


@pytest.mark.parametrize("ep_mode", [False, True])
def test_inplace_backends_compute_shared_expert_first(monkeypatch, ep_mode):
    monkeypatch.setattr(layer_module, "use_sm100_mega_moe", lambda method: ep_mode)
    layer, _ids = _layer(ep_mode)
    events = []
    x = torch.ones(3, 4)
    layer._ffn_shared_expert = _shared_recorder(events)

    def routed(**kwargs):
        events.append("routed")
        kwargs["input_tensor"].fill_(9)
        return torch.full((3, 4), 3.0)

    reductions = []
    layer._tpsp_reduce = lambda input, infer_state: reductions.append(input) or input
    experts = SimpleNamespace(quant_method=None, experts_with_topk=routed)
    result = layer._ffn(x, SimpleNamespace(is_prefill=False), SimpleNamespace(experts=experts))
    assert events == ["shared", "routed"]
    assert torch.all(result == 5)
    assert len(reductions) == (0 if ep_mode else 1)
