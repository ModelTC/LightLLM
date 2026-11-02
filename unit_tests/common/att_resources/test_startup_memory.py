from contextlib import nullcontext
from types import SimpleNamespace
import weakref

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.basemodel.attention import startup
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_infer.transformer_layer_infer import TransformerLayerInfer
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize(
    "kind", ["gqa", "int4", "int8", "static_head", "static_tensor", "mla", "packed", "packed_fp8", "packed_glm"]
)
def test_native_storage_can_be_reallocated_without_changing_logical_identity(kind):
    from unit_tests.common.att_resources.test_capacity import definition

    state = definition(kind)
    state.device = torch.device("cpu")
    state.allocate_storage(20)
    config = state.config
    references = [weakref.ref(getattr(state, name)) for name in state.storage_fields]
    state.release_storage()
    assert all(reference() is None for reference in references)
    state.allocate_storage(12)
    assert state.config is config
    assert state.capacity == 12


def startup_model(monkeypatch, *, explicit=False, peak=0, failures=0):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        startup,
        "StartupMemoryProfile",
        lambda: SimpleNamespace(stage=lambda *_: nullcontext(), peak_bytes=peak, explain=lambda: {}),
    )
    args = StartArgs(page_size=4)
    manager = AttentionStateManager(request_capacity=2)
    state = GQAFullAttentionState(GQAAttentionConfig(((0,),), 1, 8, torch.bfloat16), device="cpu")
    manager.register_state("main", state)
    manager.finish_state_registration()
    initialize = manager.initialize_storage
    manager.initialize_storage = lambda *a, **kw: initialize(*a, device="cpu", memory_limits={"cpu": 2048}, **kw)
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.args = args
    model.att_state_manager = manager
    model.is_mtp_draft_model = False
    model.max_total_token_num = 40 if explicit else None
    model.mem_fraction = 0.9
    model.torch_memory_saver = SimpleNamespace(region=lambda **_: nullcontext())
    model.req_manager = SimpleNamespace(free_all=lambda: None)
    model.layers_infer = [TransformerLayerInfer.__new__(TransformerLayerInfer)]
    model.attention_memory_demands = lambda: ()
    capacities, released = [], []

    def init_runtime():
        assert all(reference() is None for reference in released)
        capacities.append(manager.full_att_allocator.size)
        model.max_total_token_num = capacities[-1]
        view = state.get_layer_view(0)
        model.layers_infer[0].full_att_view = view
        model.graph = {1: view}
        model.bound_states = {0: view}
        model.attention_bindings = {0: view}
        if len(capacities) <= failures:
            released.extend(weakref.ref(tensor) for tensor in view)
            raise torch.OutOfMemoryError("injected after a partial graph capture")

    model.init_runtime = init_runtime
    return model, manager, capacities, released


def test_automatic_capacity_retries_after_releasing_partial_graph_and_copy_views(monkeypatch):
    model, manager, capacities, released = startup_model(monkeypatch, failures=1)
    startup.initialize_attention_runtime((model,))
    assert len(capacities) == 2 and capacities[1] < capacities[0]
    assert all(reference() is None for reference in released)
    assert manager.storage_published
    assert manager.startup_memory_attempts[0]["oom"]
    assert manager.startup_memory_attempts[1]["oom"] is None
    with pytest.raises(RuntimeError, match="published"):
        manager.release_unpublished_storage()


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_capacity_is_not_silently_reduced(monkeypatch, failure):
    model, manager, capacities, _ = startup_model(monkeypatch, explicit=True, peak=100, failures=int(failure))
    with pytest.raises(torch.OutOfMemoryError, match="capacity 40"):
        startup.initialize_attention_runtime((model,))
    assert capacities == [40]
    assert not manager.storage_published


def test_observed_graph_peak_replans_before_publication(monkeypatch):
    model, manager, capacities, _ = startup_model(monkeypatch, peak=200)
    startup.initialize_attention_runtime((model,))
    assert capacities == [40, 32]
    assert manager.memory_plan.usage("cpu") <= 2048
    assert manager.storage_published


def test_explicit_capacity_can_use_unallocated_budget_for_runtime(monkeypatch):
    model, manager, capacities, _ = startup_model(monkeypatch, explicit=True, peak=200)
    model.max_total_token_num = 16
    startup.initialize_attention_runtime((model,))
    assert capacities == [16]
    assert manager.storage_published
    assert dict(manager.memory_plan.explain()["cpu"]["allocations"])["observed_runtime_peak"] == 200


def test_retry_is_bounded_and_does_not_publish_failure(monkeypatch):
    model, manager, capacities, _ = startup_model(monkeypatch, failures=4)
    with pytest.raises(torch.OutOfMemoryError, match="startup exceeds"):
        startup.initialize_attention_runtime((model,))
    assert len(capacities) == 3
    assert not manager.storage_published


def test_distributed_oom_never_rebinds_while_peers_may_be_executing(monkeypatch):
    model, manager, capacities, _ = startup_model(monkeypatch, failures=1)

    model.init_runtime = lambda: (_ for _ in ()).throw(torch.OutOfMemoryError("peer may still be in a collective"))
    monkeypatch.setattr(startup, "_distributed", lambda: True)
    with pytest.raises(torch.OutOfMemoryError, match="peer"):
        startup.initialize_attention_runtime((model,))
    assert not manager.storage_published


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_memory_profile_observes_freed_transients_and_graph_pool():
    torch.cuda.empty_cache()
    profile = startup.StartupMemoryProfile()
    with profile.stage("transient"):
        transient = torch.ones(16 * 1024 ** 2, dtype=torch.uint8, device="cuda")
        del transient
    with profile.stage("graph"):
        graph = torch.cuda.CUDAGraph()
        inputs = torch.ones(2 * 1024 ** 2, dtype=torch.uint8, device="cuda")
        with torch.cuda.graph(graph):
            output = inputs + 1
        graph.replay()
    assert profile.peak_bytes >= 16 * 1024 ** 2
    assert [item.stage for item in profile.observations] == ["transient", "graph"]
    assert profile.observations[1].allocated_bytes > profile.initial_allocated
    assert torch.all(output == 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_replanning_recaptures_against_new_kv_storage(monkeypatch):
    profile_class = startup.StartupMemoryProfile
    synchronize, empty_cache = torch.cuda.synchronize, torch.cuda.empty_cache
    model, manager, capacities, released = startup_model(monkeypatch)
    monkeypatch.setattr(startup, "StartupMemoryProfile", profile_class)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    manager.initialize_storage = lambda *a, **kw: AttentionStateManager.initialize_storage(
        manager, *a, device="cuda", memory_limits={"cpu": 64 * 1024 ** 2, "cuda": 32 * 1024 ** 2}, **kw
    )
    state = manager.get_full_attention_states()[0]

    def capture():
        assert all(reference() is None for reference in released)
        capacities.append(state.capacity)
        model.max_total_token_num = state.capacity
        view = state.k_storage[0, :4]
        view.fill_(len(capacities))
        graph = torch.cuda.CUDAGraph()
        transient = torch.empty(8 * 1024 ** 2, dtype=torch.uint8, device="cuda")
        del transient
        with torch.cuda.graph(graph):
            output = view + 1
        graph.replay()
        model.graph = {1: (graph, view, output)}
        model.layers_infer[0].full_att_view = view
        released.extend((weakref.ref(state.k_storage), weakref.ref(output)))

    model.init_runtime = capture
    startup.initialize_attention_runtime((model,))
    assert 2 <= len(capacities) <= 3
    assert capacities[-1] < capacities[0]
    graph, view, output = model.graph[1]
    view.fill_(7)
    graph.replay()
    assert torch.all(output == 8)
    assert all(reference() is None for reference in released[:-2])
    assert manager.storage_published


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autotune_oom_restores_layers_phase_and_request_allocations(monkeypatch):
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.layers_num = 24
    model.batch_max_tokens = 4
    model.autotune_layers = lambda: 1
    model._gen_special_model_input = lambda _: {}
    releases = []
    model.req_manager = SimpleNamespace(
        alloc=lambda: 0,
        req_to_token_indexs=torch.zeros(2, 4, dtype=torch.int32, device="cuda"),
        free_all=lambda: releases.append("request"),
    )
    model.att_state_manager = SimpleNamespace(
        full_att_allocator=SimpleNamespace(
            page_size=1,
            alloc=lambda count: torch.arange(count, dtype=torch.int32),
            free_all=lambda: releases.append("kv"),
        )
    )

    def forward(_):
        assert model.layers_num == 1
        assert Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
        raise torch.OutOfMemoryError("injected warmup allocation failure")

    model.forward = forward
    with pytest.raises(torch.OutOfMemoryError, match="warmup allocation"):
        model._autotune_warmup()
    assert model.layers_num == 24
    assert not Autotuner.is_autotune_warmup()
    assert releases == ["request", "kv"]
