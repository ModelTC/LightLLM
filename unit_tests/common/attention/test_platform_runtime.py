from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase, ExecutionRequirements, GraphBinding
from lightllm.common.attention.backends._workspace import AttentionWorkspace
from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend
from lightllm.common.attention.backends.dense.torch_npu.backend import TorchNpuAttBackend
from lightllm.common.attention.backends.dense.triton.musa import MusaTritonAttBackend
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAFullAttentionState, GQAAttentionConfig
from lightllm.common.basemodel.graph import GraphExecution
from lightllm.platform import DeviceCaps, DeviceRuntime, Platform, get_device_runtime
from unit_tests.common.attention.helpers import make_spec, make_batch, make_plan, model_stub, bind


def test_reject_other_platform_before_loading_cuda_extension(monkeypatch):
    monkeypatch.setattr("importlib.import_module", lambda *args: pytest.fail("loaded a CUDA dependency"))
    caps = DeviceCaps(Platform.MUSA, torch.device("cpu"), supports_graph=True)
    result = Fa3AttBackend.check_environment(caps)
    assert not result.supported
    assert "musa" in result.reason


def test_runtime_resolves_host_platform_without_querying_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("queried CUDA for a CPU runtime"))
    assert get_device_runtime(platform="cpu").device == torch.device("cpu")
    with pytest.raises(ValueError, match="cannot operate"):
        get_device_runtime("cuda:0", platform="cpu")
    with pytest.raises(ValueError, match="cannot operate"):
        get_device_runtime("cpu", platform="cuda")


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


@pytest.mark.parametrize(
    "backend,platform", [(MusaTritonAttBackend, Platform.MUSA), (TorchNpuAttBackend, Platform.ASCEND)]
)
def test_native_adapters_reject_modes_before_allocation(backend, platform):
    spec = make_spec(head_size=64, value_head_size=64)
    spec = replace(spec, cache=replace(spec.cache, page_size=32, key_dim=64, value_dim=64))
    caps = DeviceCaps(platform, torch.device("cpu"), supports_graph=True)
    execution = ExecutionRequirements(graph_phases=frozenset({AttentionPhase.DECODE}))
    assert backend.check_support(spec, caps, execution).supported
    for requirements in (
        replace(execution, graph_phases=frozenset({AttentionPhase.PREFILL})),
        replace(execution, microbatch_count=2),
        replace(execution, transfer_paths=frozenset({"pd"})),
        replace(execution, expert_parallel=True),
    ):
        phase = AttentionPhase.PREFILL if AttentionPhase.PREFILL in requirements.graph_phases else spec.phase
        assert not backend.check_support(replace(spec, phase=phase), caps, requirements).supported
    assert not backend.check_support(replace(spec, use_mtp=True), caps, execution).supported


def test_fia_host_snapshots_and_workspace_are_prepared_before_capture(monkeypatch):
    import sys
    from lightllm.common.attention.api.inputs import DenseInputs, DenseParams

    model = model_stub(page_size=32)
    model.device_runtime = DeviceRuntime("cpu")
    backend = TorchNpuAttBackend(model)
    batch = make_batch((7, 13))
    plan = make_plan(backend, batch, capture=True, max_length=32, prepare=True)
    host = plan.host_kv_lengths
    backend.prepare(plan, make_batch((9, 5), req_to_token_indexs=batch.req_to_token_indexs))
    assert plan.host_kv_lengths is host and host == [9, 5]
    assert plan.host_q_lengths == [1, 1]

    query = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
    cache = torch.zeros(32, 2, 64, dtype=query.dtype)
    layer = bind(backend, plan, DenseParams(cache, cache), head_size=64, value_head_size=64)
    monkeypatch.setattr(backend.runtime, "is_capturing", lambda: True)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    with pytest.raises(RuntimeError, match="warmup before capture"):
        layer(plan, DenseInputs(query))

    monkeypatch.setattr(backend.runtime, "is_capturing", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            _npu_fused_infer_attention_score_get_max_workspace=lambda **kwargs: torch.empty(
                backend.workspace_bytes + 1, dtype=torch.uint8, device="meta"
            )
        ),
    )
    with pytest.raises(ValueError, match="exceeding reservation"):
        layer(plan, DenseInputs(query))


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
    backend = SimpleNamespace(bind_graph=lambda plan: Binding())
    state = SimpleNamespace(
        input_ids=torch.zeros(1),
        att_state_manager=manager,
        attention_groups=(SimpleNamespace(backend_for=lambda phase: backend),),
        attention_plans={
            0: SimpleNamespace(
                capacity=SimpleNamespace(phase=AttentionPhase.DECODE), workspace=AttentionWorkspace(runtime=runtime)
            )
        },
        copy_for_cuda_graph=lambda source: events.append("copy"),
    )
    graph = SimpleNamespace(replay=submit, reset=lambda: events.append("graph_reset"))
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
        attention_plans={},
        attention_groups=(),
        copy_for_cuda_graph=lambda source: pytest.fail("copied into a stale graph"),
    )
    graph = SimpleNamespace(replay=lambda: pytest.fail("replayed a stale graph"), reset=lambda: None)
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

    def bind_graph(plan):
        if plan.fail:
            raise ValueError("unsupported graph task")
        return Binding()

    manager = AttentionStateManager(request_capacity=2)
    backend = SimpleNamespace(bind_graph=bind_graph)
    state = SimpleNamespace(
        att_state_manager=manager,
        attention_groups=(SimpleNamespace(backend_for=lambda phase: backend),) * 2,
        attention_plans={
            i: SimpleNamespace(
                fail=bool(i), capacity=SimpleNamespace(phase=AttentionPhase.DECODE), workspace=AttentionWorkspace("cpu")
            )
            for i in range(2)
        },
    )
    graph = SimpleNamespace(reset=lambda: events.append("graph_reset"))
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
