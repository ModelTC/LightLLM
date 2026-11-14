from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase, CaptureHooks
from lightllm.common.attention.api.workspace import AttentionWorkspace
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att.gqa.gqa_state import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.attention.session import AttentionSession
from lightllm.common.basemodel.graph import CapturedGraph
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
    first, second = pool.borrow_for_capture(), pool.borrow_for_capture()
    old = first.scratch("scratch", 8, torch.int32)
    assert second.scratch("scratch", 8, torch.int32).data_ptr() == old.data_ptr()
    first.freeze()
    grown = pool.scratch("scratch", 64, torch.int32)
    old.fill_(17)
    grown.fill_(29)
    assert first.scratch("scratch", 8, torch.int32).tolist() == [17] * 8
    with pytest.raises(RuntimeError, match="cannot grow"):
        first.scratch("scratch", 64, torch.int32)
    assert AttentionWorkspace("cpu").scratch("scratch", 8, torch.int32).data_ptr() != old.data_ptr()


def test_graph_orders_host_reuse_submission_update_and_destruction():
    events = []
    pending = {"submitted": False, "updated": False}

    class Runtime(DeviceRuntime):
        def synchronize(self):
            assert not pending["submitted"] or pending["updated"], "waited for a graph before submitting its updates"
            events.append("synchronize")

    class Binding(CaptureHooks):
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
    backend = SimpleNamespace(create_capture_hooks=lambda context: Binding())
    state = SimpleNamespace(
        input_ids=torch.zeros(1),
        att_state_manager=manager,
        attention_groups=({AttentionPhase.DECODE: backend},),
        attention_contexts={
            0: SimpleNamespace(
                capacity=SimpleNamespace(phase=AttentionPhase.DECODE), workspace=AttentionWorkspace(runtime=runtime)
            )
        },
        graph_input_copies=lambda source: [(SimpleNamespace(copy_=lambda *a, **kw: events.append("copy")), None)],
    )
    graph = SimpleNamespace(replay=submit, reset=lambda: events.append("graph_reset"))
    state.attention = AttentionSession()
    state.attention.groups = state.attention_groups
    state.attention.contexts = state.attention_contexts
    state.attention.plan_replay = lambda source: events.append("plan")
    execution = CapturedGraph(graph, (state,), ("result",), runtime=runtime)
    assert execution.replay(state) == "result"
    assert events == ["host_ready", "copy", "plan", "submit", "update"]
    manager.close_graphs()
    execution.close()
    assert events[-4:] == ["host_ready", "synchronize", "binding_close", "graph_reset"]
    assert execution.infer_states == ()
    with pytest.raises(RuntimeError, match="closed"):
        execution.replay(state)


def test_storage_generation_invalidates_graph_before_copy():
    manager = AttentionStateManager(request_capacity=2)
    state = SimpleNamespace(
        input_ids=torch.zeros(1),
        att_state_manager=manager,
        attention_contexts={},
        attention_groups=(),
        graph_input_copies=lambda source: pytest.fail("copied into a stale graph"),
    )
    graph = SimpleNamespace(replay=lambda: pytest.fail("replayed a stale graph"), reset=lambda: None)
    state.attention = AttentionSession()
    state.attention.groups = state.attention_groups
    state.attention.contexts = state.attention_contexts
    execution = CapturedGraph(graph, (state,), runtime=DeviceRuntime("cpu"))
    manager.storage_generation += 1
    with pytest.raises(RuntimeError, match="storage changed"):
        execution.replay(state)
    execution.close()


@pytest.mark.parametrize("invalid", ["role", "shape", "missing_hidden", "same_pointer_shape"])
def test_graph_validates_all_lanes_before_copy_or_plan(monkeypatch, invalid):
    from lightllm.common.attention.api import BackendEnv, ContextCapacity
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from unit_tests.common.attention.helpers import StubBackend, make_batch, make_spec

    runtime = DeviceRuntime("cpu")
    manager = AttentionStateManager(request_capacity=2)
    env = BackendEnv(
        runtime=runtime,
        hold_req_id=7,
        draft_step=0,
        max_seq_len=16,
        tp_size=1,
        alloc_tensor=torch.empty,
    )
    backend = StubBackend(env, make_spec())
    groups = ({AttentionPhase.DECODE: backend},)
    plans = []
    monkeypatch.setattr(backend, "plan_host", lambda context: plans.append(context))
    capacity = ContextCapacity(
        phase=AttentionPhase.DECODE, batch_size=2, token_num=2, max_kv_seq_len=16, max_q_seq_len=1
    )

    def make_state(batch, capture):
        state = InferStateInfo()
        state.att_state_role = "main"
        state.att_state_manager = manager
        state.input_ids = torch.zeros(2, dtype=torch.int64) if capture else torch.ones(2, dtype=torch.int64)
        with state.attention.prepare(
            batch,
            groups,
            capacity,
            workspace=lambda group_id: AttentionWorkspace(runtime=runtime),
            capture=capture,
            replay=not capture,
        ):
            pass
        return state

    targets = [make_state(make_batch(), True) for _ in range(2)]
    sources = [
        make_state(make_batch((9, 7), req_to_token_indexs=target.attention.batch.req_to_token_indexs), False)
        for target in targets
    ]
    if invalid == "role":
        sources[1].att_state_role = "draft0"
    elif invalid == "missing_hidden":
        targets[1].mtp_draft_input_hiddens = torch.ones(2, 4)
    elif invalid == "same_pointer_shape":
        sources[1].input_ids = targets[1].input_ids.view(1, 2)
    else:
        sources[1].input_ids = torch.ones(3, dtype=torch.int64)
    plans.clear()
    graph = SimpleNamespace(replay=lambda: pytest.fail("submitted invalid lanes"), reset=lambda: None)
    execution = CapturedGraph(graph, targets, runtime=runtime)
    try:
        with pytest.raises(ValueError, match="model role|shape, dtype or device|changed presence"):
            execution.replay(*sources)
        assert targets[0].input_ids.tolist() == [0, 0]
        assert targets[0].attention.batch.b_seq_len.tolist() == [5, 2]
        assert plans == []
    finally:
        execution.close()


def test_graph_binding_failure_releases_previously_bound_resources():
    events = []

    class Binding(CaptureHooks):
        def close(self):
            events.append("binding_close")

    def create_capture_hooks(context):
        if context.fail:
            raise ValueError("unsupported graph task")
        return Binding()

    manager = AttentionStateManager(request_capacity=2)
    backend = SimpleNamespace(create_capture_hooks=create_capture_hooks)
    state = SimpleNamespace(
        att_state_manager=manager,
        attention_groups=({AttentionPhase.DECODE: backend},) * 2,
        attention_contexts={
            i: SimpleNamespace(
                fail=bool(i), capacity=SimpleNamespace(phase=AttentionPhase.DECODE), workspace=AttentionWorkspace("cpu")
            )
            for i in range(2)
        },
    )
    graph = SimpleNamespace(reset=lambda: events.append("graph_reset"))
    state.attention = AttentionSession()
    state.attention.groups = state.attention_groups
    state.attention.contexts = state.attention_contexts
    with pytest.raises(ValueError, match="unsupported graph task"):
        CapturedGraph(graph, (state,), runtime=DeviceRuntime("cpu"))
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
