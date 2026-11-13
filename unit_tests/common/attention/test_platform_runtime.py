from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase, GraphBinding
from lightllm.common.attention.backends._workspace import AttentionWorkspace
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.basemodel.attention_runtime import AttentionRuntime
from lightllm.common.basemodel.graph import GraphExecution
from lightllm.platform import DeviceRuntime, get_device_runtime


def test_runtime_resolves_host_platform_without_querying_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("queried CUDA for a CPU runtime"))
    assert get_device_runtime(platform="cpu").device == torch.device("cpu")
    with pytest.raises(ValueError, match="cannot operate"):
        get_device_runtime("cuda:0", platform="cpu")
    with pytest.raises(ValueError, match="cannot operate"):
        get_device_runtime("cpu", platform="cuda")


def test_runtime_rejects_unimplemented_hardware():
    with pytest.raises(ValueError, match="CUDA and CPU"):
        get_device_runtime(platform="musa")
    with pytest.raises(ValueError, match="CUDA and CPU"):
        get_device_runtime("npu:0")


def test_explicit_current_device_is_not_overridden_by_worker_default(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert get_device_runtime("cuda").device == torch.device("cuda:1")
    assert get_device_runtime("cuda:2").device == torch.device("cuda:2")
    assert get_device_runtime().device == torch.device("cuda:0")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_gqa_storage_allocation_uses_explicit_device_context(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    config = GQAAttentionConfig(logical_layer_groups=((0,),), global_head_num=2, head_dim=8, dtype=torch.float32)
    with torch.cuda.device(1):
        state = GQAFullAttentionState(config, device="cuda", page_size=4)
        state.allocate_storage(8)
    assert state.k_storage.device == state.v_storage.device == torch.device("cuda:1")


def test_captured_workspace_keeps_address_when_pool_grows():
    pool = AttentionWorkspace("cpu")
    first, second = pool.fork(), pool.fork()
    old = first.get("scratch", 8, torch.int32)
    assert second.get("scratch", 8, torch.int32).data_ptr() == old.data_ptr()
    first.freeze()
    grown = pool.get("scratch", 64, torch.int32)
    old.fill_(17)
    grown.fill_(29)
    assert first.get("scratch", 8, torch.int32).tolist() == [17] * 8
    with pytest.raises(RuntimeError, match="cannot grow"):
        first.get("scratch", 64, torch.int32)
    assert AttentionWorkspace("cpu").get("scratch", 8, torch.int32).data_ptr() != old.data_ptr()


def test_graph_orders_host_reuse_submission_update_and_destruction():
    events = []
    pending = {"submitted": False, "updated": False}

    class Runtime(DeviceRuntime):
        def synchronize(self):
            assert not pending["submitted"] or pending["updated"], "waited for a graph before submitting its updates"
            events.append("synchronize")

    class Binding(GraphBinding):
        def wait_for_update(self):
            events.append("host_ready")

        def after_submit(self):
            assert pending["submitted"]
            pending["updated"] = True
            events.append("update")
            return ()

        def close(self):
            events.append("binding_close")

    def submit():
        pending["submitted"] = True
        events.append("submit")

    runtime = Runtime("cpu")
    manager = AttentionStateManager(request_capacity=2)
    backend = SimpleNamespace(bind_graph=lambda context: Binding())
    state = SimpleNamespace(
        input_ids=torch.zeros(1),
        att_state_manager=manager,
        attention_groups=(SimpleNamespace(backend_for=lambda phase: backend),),
        attention_contexts={
            0: SimpleNamespace(
                capacity=SimpleNamespace(phase=AttentionPhase.DECODE), workspace=AttentionWorkspace(runtime=runtime)
            )
        },
        copy_for_cuda_graph=lambda source: events.append("copy"),
    )
    graph = SimpleNamespace(replay=submit, reset=lambda: events.append("graph_reset"))
    state.attention_runtime = AttentionRuntime()
    state.attention_runtime.groups = state.attention_groups
    state.attention_runtime.contexts = state.attention_contexts
    execution = GraphExecution(graph, (state,), ("result",), runtime=runtime)
    assert execution.replay(state) == "result"
    assert events == ["host_ready", "copy", "submit", "update"]
    manager.close_graphs()
    execution.close()
    assert events[-4:] == ["host_ready", "synchronize", "binding_close", "graph_reset"]
    assert execution.states == ()
    with pytest.raises(RuntimeError, match="closed"):
        execution.replay(state)


def test_storage_generation_invalidates_graph_before_copy():
    manager = AttentionStateManager(request_capacity=2)
    state = SimpleNamespace(
        input_ids=torch.zeros(1),
        att_state_manager=manager,
        attention_contexts={},
        attention_groups=(),
        copy_for_cuda_graph=lambda source: pytest.fail("copied into a stale graph"),
    )
    graph = SimpleNamespace(replay=lambda: pytest.fail("replayed a stale graph"), reset=lambda: None)
    state.attention_runtime = AttentionRuntime()
    state.attention_runtime.groups = state.attention_groups
    state.attention_runtime.contexts = state.attention_contexts
    execution = GraphExecution(graph, (state,), runtime=DeviceRuntime("cpu"))
    manager.storage_generation += 1
    with pytest.raises(RuntimeError, match="storage changed"):
        execution.replay(state)
    execution.close()


def test_graph_binding_failure_releases_previously_bound_resources():
    events = []

    class Binding(GraphBinding):
        def close(self):
            events.append("binding_close")

    def bind_graph(context):
        if context.fail:
            raise ValueError("unsupported graph task")
        return Binding()

    manager = AttentionStateManager(request_capacity=2)
    backend = SimpleNamespace(bind_graph=bind_graph)
    state = SimpleNamespace(
        att_state_manager=manager,
        attention_groups=(SimpleNamespace(backend_for=lambda phase: backend),) * 2,
        attention_contexts={
            i: SimpleNamespace(
                fail=bool(i), capacity=SimpleNamespace(phase=AttentionPhase.DECODE), workspace=AttentionWorkspace("cpu")
            )
            for i in range(2)
        },
    )
    graph = SimpleNamespace(reset=lambda: events.append("graph_reset"))
    state.attention_runtime = AttentionRuntime()
    state.attention_runtime.groups = state.attention_groups
    state.attention_runtime.contexts = state.attention_contexts
    with pytest.raises(ValueError, match="unsupported graph task"):
        GraphExecution(graph, (state,), runtime=DeviceRuntime("cpu"))
    manager.close_graphs()
    assert events == ["binding_close", "graph_reset"]


def test_state_layout_and_native_write_preserve_physical_rows():
    config = GQAAttentionConfig(logical_layer_groups=((0, 3), (7,)), global_head_num=2, head_dim=8, dtype=torch.float32)
    state = GQAFullAttentionState(config, device="cpu", page_size=4)
    state.require_zero_initialization()
    state.allocate_storage(12)
    assert not state.k_storage.count_nonzero()
    values = torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8)
    state.write_full_att_kv(3, torch.tensor([2, 5], dtype=torch.int32), values)
    torch.testing.assert_close(state.get_layer_view(0)[0][[2, 5]], values[:, :2])
    state.copy_full_att_tokens(torch.tensor([2, 5]), torch.tensor([7, 9]))
    torch.testing.assert_close(state.get_layer_view(3)[1][[7, 9]], values[:, 2:])
    assert not state.get_layer_view(7)[0].count_nonzero()
    relocated = type("ExternalState", (GQAFullAttentionState,), {"__module__": "external.provider"})
    assert relocated(config, device="cpu").describe_layout() == state.describe_layout()


def test_runtime_registration_keeps_platform_validation_in_adapter(monkeypatch):
    import lightllm.platform as platform

    monkeypatch.setattr(platform, "_runtime_factories", {})
    calls = []
    adapter = object()
    platform.register_device_runtime("musa", lambda device: calls.append(device) or adapter)
    assert platform.get_device_runtime("musa:2", platform="musa") is adapter
    assert calls == ["musa:2"]
    with pytest.raises(ValueError, match="already registered"):
        platform.register_device_runtime("musa", lambda device: adapter)
