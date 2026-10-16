"""EPLB routing across the shared-expert and split ElasticBuffer interfaces."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight as weight_module
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    EPLBState,
    ExpertParallelState,
)
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import deepgemm_impl as impl_module
from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_topk


def make_impl():
    state = EPLBState(
        num_redundant_experts_per_rank=1,
        logical_to_physical_map=torch.tensor([[2, 4], [3, 5]], dtype=torch.int32),
        logical_replica_count=torch.ones(2, dtype=torch.int32),
        route_counter=torch.zeros((4, 2), dtype=torch.int64),
        recording=True,
    )
    impl = impl_module.FuseMoeDeepGEMM(
        2, 0, 1.0, SimpleNamespace(), expert_parallel_state=ExpertParallelState(2, 2, state)
    )
    return impl


def test_external_topk_capture_precedes_mapping_and_preserves_shared_callback_and_group(monkeypatch):
    impl = make_impl()
    weight = weight_module.FusedMoeWeight.__new__(weight_module.FusedMoeWeight)
    weight.fuse_moe_impl, weight.layer_num_, weight.w13, weight.w2 = impl, 0, object(), object()
    logical = torch.tensor([[0, 1]])
    original = logical.clone()
    events = []
    group = object()
    shared = lambda: None

    def capture(ids):
        events.append("capture")
        assert torch.equal(ids, original)

    def map_ids(ids, state, **kwargs):
        events.append("map")
        assert kwargs["is_prefill"] is True
        return ids + 2

    def execute(**kwargs):
        events.append("execute")
        assert kwargs["topk_ids"].tolist() == [[2, 3]]
        assert kwargs["shared_expert_func"] is shared
        assert kwargs["dist_group"] is group
        return "output"

    monkeypatch.setattr(weight_module, "get_moe_capture_callback", lambda *_: capture)
    monkeypatch.setattr(grouped_topk, "map_eplb_topk", map_ids)
    monkeypatch.setattr(impl, "_fused_experts", execute)
    assert (
        weight.experts_with_topk(
            torch.empty((1, 4)),
            torch.ones((1, 2)),
            logical,
            True,
            infer_state=SimpleNamespace(dist_group=group),
            shared_expert_func=shared,
        )
        == "output"
    )
    assert events == ["capture", "map", "execute"]
    assert torch.equal(logical, original)


def test_decode_maps_external_routes_once_and_does_not_remap_internal_routes(monkeypatch):
    impl = make_impl()
    maps, dispatches = [], []
    logical = torch.tensor([[0, 1]])
    physical = torch.tensor([[2, 3]])

    def map_ids(ids, state, **kwargs):
        maps.append(kwargs)
        return physical

    def dispatch(inputs, **kwargs):
        dispatches.append(kwargs)
        return "recv", None, "weights", "handle", SimpleNamespace(current_stream_wait=lambda: None)

    monkeypatch.setattr(grouped_topk, "map_eplb_topk", map_ids)
    monkeypatch.setattr(impl_module, "quantize_fused_experts_input", lambda *args: "quantized")
    monkeypatch.setattr(impl_module, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 16)
    monkeypatch.setattr(impl_module, "get_mk_alignment_for_contiguous_layout", lambda: 128)
    monkeypatch.setattr(impl_module, "get_ep_num_sms", lambda **kwargs: 8)
    monkeypatch.setattr(impl_module.dist_group_manager, "ep_buffer", SimpleNamespace(dispatch=dispatch))
    monkeypatch.setattr(impl, "_select_eplb_experts", lambda **kwargs: (torch.ones((1, 2)), physical, logical))
    hidden = torch.empty((1, 4))
    impl.decode_dispatch_with_topk(hidden, object(), logical, torch.ones((1, 2)))
    impl.decode_dispatch(hidden, torch.empty((1, 2)), None, object(), False, 2, False, 0, 0, "softmax")
    assert len(maps) == 1 and maps[0]["is_prefill"] is False
    assert len(dispatches) == 2
    assert all(call["topk_idx"] is physical and call["num_experts"] == 4 for call in dispatches)


def test_split_combine_waits_before_synchronizing_the_correct_tp_group(monkeypatch):
    impl = make_impl()
    impl.eplb.tp_output_sync_src = 0
    group, output = object(), torch.zeros((1, 4))
    events = []
    event = SimpleNamespace(current_stream_wait=lambda: events.append("wait"))
    monkeypatch.setattr(impl_module, "_record_compute_stream", lambda *args: None)
    monkeypatch.setattr(impl_module, "get_ep_num_sms", lambda **kwargs: 8)
    monkeypatch.setattr(
        impl_module.dist_group_manager,
        "ep_buffer",
        SimpleNamespace(combine=lambda *args, **kwargs: (output, None, event)),
    )
    import lightllm.distributed as distributed

    def broadcast(tensor, src, group):
        assert tensor is output
        events.append(("broadcast", src, group))

    monkeypatch.setattr(distributed, "broadcast", broadcast)
    result, hook = impl.prefill_combine(torch.empty((1, 4)), object(), dist_group=group)
    assert result is output and not events
    hook()
    assert events == ["wait", ("broadcast", 0, group)]


def test_split_prefill_keeps_one_layout_generation_for_both_microbatches(monkeypatch):
    impl = make_impl()
    generation = []
    impl.eplb.migration_poll = lambda: generation.append(len(generation) + 1)

    def select(**kwargs):
        return torch.ones((1, 2)), torch.full((1, 2), generation[-1], dtype=torch.long), None

    monkeypatch.setattr(grouped_topk, "triton_grouped_topk_eplb", select)
    monkeypatch.setattr(impl_module, "quantize_fused_experts_input", lambda *args: "quantized")
    args = (torch.empty((1, 4)), torch.empty((1, 2)), None, object(), False, 2, False, 0, 0, "softmax")
    _, first, _ = impl.select_experts_and_quant_input(*args)
    _, second, _ = impl.select_experts_and_quant_input(*args, poll_migration=False)
    assert generation == [1]
    assert torch.equal(first, second)


def test_empty_external_prefill_advances_sampling_without_launching_a_kernel(monkeypatch):
    impl = make_impl()
    monkeypatch.setattr(grouped_topk, "_map_eplb_topk_kernel", None)
    result = grouped_topk.map_eplb_topk(torch.empty((0, 2), dtype=torch.long), impl.eplb, is_prefill=True)
    assert result.shape == (0, 2)
    assert impl.eplb.recorded_sample_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_prefill", [False, True])
def test_external_mapping_matches_fused_eplb_selection(is_prefill):
    logits = torch.tensor([[4.0, 1.0], [1.0, 4.0], [3.0, 2.0]], device="cuda")
    hidden = torch.empty((3, 4), device="cuda")
    impl = make_impl()
    impl.eplb.logical_to_physical_map = impl.eplb.logical_to_physical_map.cuda()
    impl.eplb.logical_replica_count = impl.eplb.logical_replica_count.cuda()
    impl.eplb.route_counter = torch.zeros((4, 2, 4), dtype=torch.int64, device="cuda")
    kwargs = dict(
        hidden_states=hidden,
        gating_output=logits,
        correction_bias=None,
        topk=2,
        renormalize=False,
        num_expert_group=0,
        topk_group=0,
        scoring_func="softmax",
        logical_to_physical_map=impl.eplb.logical_to_physical_map,
        logical_replica_count=impl.eplb.logical_replica_count,
        expert_counter=impl.eplb.route_counter,
        sample_index=0,
        record_load=is_prefill,
        use_grouped_topk=False,
        return_logical_ids=True,
        use_replicas=is_prefill,
    )
    _, expected, logical = grouped_topk.triton_grouped_topk_eplb(**kwargs)
    expected_counts = impl.eplb.route_counter[0].clone()
    impl.eplb.route_counter.zero_()
    actual = grouped_topk.map_eplb_topk(logical, impl.eplb, is_prefill=is_prefill)
    assert torch.equal(actual, expected)
    assert torch.equal(impl.eplb.route_counter[0], expected_counts)
