import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.moe_route_info_manager import MoeRouteInfoManager, get_moe_capture_callback
from lightllm.core.runner.state import BatchStateFactory
from unit_tests.common.execution.test_batch_runner import prefill_input


def make_factory(binding, indices):
    return BatchStateFactory(
        state_type=InferStateInfo,
        hidden_collector=SimpleNamespace(new_instance=lambda: None),
        requests=None,
        state_store=None,
        role="main",
        is_draft=False,
        return_all_prompt_logics=False,
        group_for_lane=lambda lane: lane,
        select_indices=lambda batch: indices.clone(),
        route_capture=binding,
    )


@pytest.mark.parametrize("master,enabled", [(True, True), (False, True), (True, False)])
def test_backend_binds_recording_before_runtime(master, enabled, monkeypatch):
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend

    events, binding = [], object()
    manager = SimpleNamespace(
        init_capture_buffer=lambda **kw: events.append(kw["kv_cache_size"]), bind_capture=lambda: binding
    )
    monkeypatch.setattr(MoeRouteInfoManager, "get_instance", lambda: manager)
    backend = ModeBackend.__new__(ModeBackend)
    backend.is_master_in_dp = master
    backend.args = SimpleNamespace(enable_return_routed_experts=enabled)
    backend.model = SimpleNamespace(att_state_manager=SimpleNamespace(full_att_allocator=SimpleNamespace(size=32)))
    draft = SimpleNamespace()
    backend.draft_models = [draft]
    backend._init_route_capture()
    assert backend.model.route_capture is (binding if master and enabled else None)
    assert events == ([32] if master and enabled else [])
    assert not hasattr(draft, "route_capture")


def test_batch_callbacks_keep_separate_token_views_without_global_lookup(monkeypatch):
    monkeypatch.setattr(MoeRouteInfoManager, "get_instance", lambda: pytest.fail("runtime manager lookup"))
    captured = []
    binding = SimpleNamespace(bind_batch=lambda indices: {7: lambda ids: captured.append((indices, ids))})
    factory = make_factory(binding, torch.tensor([2, 3]))
    first, second = [factory.create(prefill_input(2), lane) for lane in (0, 1)]
    second.mem_index.add_(4)
    for state in (first, second):
        get_moe_capture_callback(state, 7)(state.microbatch_index)
        assert get_moe_capture_callback(state, 3) is None
    assert [(indices.tolist(), lane) for indices, lane in captured] == [([2, 3], 0), ([6, 7], 1)]


@pytest.mark.parametrize("prefill", [False, True])
def test_graph_rejects_replaced_recording_storage(prefill):
    first, second = InferStateInfo(), InferStateInfo()
    first.route_capture, second.route_capture = object(), object()
    copy = first.copy_for_prefill_cuda_graph if prefill else first.copy_for_cuda_graph
    with pytest.raises(ValueError, match="route recording binding changed"):
        copy(second)


@pytest.mark.parametrize("prefill", [False, True])
def test_legacy_ep_overlap_records_logical_ids_before_dispatch(prefill, monkeypatch):
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl import deepgemm_impl

    impl = deepgemm_impl.FuseMoeDeepGEMM.__new__(deepgemm_impl.FuseMoeDeepGEMM)
    logical, physical, scores, inputs = object(), object(), object(), object()
    impl._select_experts = lambda **kw: (scores, physical, logical)
    impl.low_latency_dispatch_with_topk = lambda **kw: kw["topk_idx"]
    captured = []
    kwargs = dict(
        hidden_states=inputs,
        router_logits=object(),
        e_score_correction_bias=None,
        use_grouped_topk=False,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        topk_group=1,
        n_group=1,
        scoring_func="softmax",
        capture=captured.append,
    )
    if prefill:
        physical = torch.tensor([[8, 9]])
        monkeypatch.setattr(deepgemm_impl, "quantize_fused_experts_input", lambda *args: inputs)
        impl.quant_method = None
        _, routed, _ = impl.select_experts_and_quant_input(**kwargs, w13=None)
        assert routed is physical
    else:
        assert impl.low_latency_dispatch(**kwargs) is physical
    assert captured == [logical]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dtype_id", [1, 2])
def test_bound_recording_graph_updates_slots_and_keeps_storage_alive(prefill, dtype_id, monkeypatch):
    manager = MoeRouteInfoManager(2, 2, dtype_id, {3: 0, 7: 1})
    manager.init_capture_buffer(16)
    binding, owner = manager.bind_capture(), weakref.ref(manager)
    del manager
    assert owner() is None
    monkeypatch.setattr(MoeRouteInfoManager, "get_instance", lambda: pytest.fail("runtime manager lookup"))
    indices = torch.tensor([2, 3, 16, -1], device="cuda")
    factory = make_factory(binding, indices)
    first, second = [factory.create(prefill_input(4), lane) for lane in (0, 1)]
    second.mem_index[:2].add_(4)
    # The routed columns can be a view preceding appended shared experts.
    ids = torch.tensor([[1, 2, 99], [3, 4, 99], [5, 6, 99], [7, 8, 99]], device="cuda")
    if dtype_id == 2:
        ids[:, :2].add_(300)
    callbacks = [get_moe_capture_callback(state, layer) for state, layer in ((first, 3), (second, 7))]
    for capture in callbacks:
        capture(ids[:, :2])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for capture in callbacks:
            capture(ids[:, :2])
    for state, rows in ((first, [4, 5]), (second, [8, 9])):
        new = factory.create(prefill_input(4))
        new.mem_index[:2].copy_(torch.tensor(rows, device="cuda"))
        copy = state.copy_for_prefill_cuda_graph if prefill else state.copy_for_cuda_graph
        copy(new)
    torch.cuda.synchronize()
    binding.buffer.zero_()
    ids[:, :2].add_(10)
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.zeros_like(binding.buffer)
    for rows, slot in (([4, 5], 0), ([8, 9], 1)):
        expected[rows, slot] = ids[:2, :2].cpu().to(expected.dtype)
    torch.testing.assert_close(binding.buffer, expected, rtol=0, atol=0)
