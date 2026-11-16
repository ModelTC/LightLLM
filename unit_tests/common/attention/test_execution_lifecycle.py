from types import SimpleNamespace


import pytest


import torch


from lightllm.common.attention.api import AttentionPhase


from lightllm.common.attention.api.graph import TensorCopyPlan, tensor_input_copy, validate_copy_plan


from lightllm.common.attention_state import AttentionStateManager


from lightllm.common.basemodel.attention import ModelAttention, ModelBinding


from lightllm.common.basemodel.graph import CapturedGraph


from lightllm.common.basemodel.infer_struct import InferStateInfo


from lightllm.common.execution import ExecutionScope


from lightllm.platform import DeviceRuntime


from unit_tests.common.attention.helpers import StubBackend, make_batch, make_env, make_spec


def make_state(batch=None, manager=None, lane=0):
    state = InferStateInfo()
    batch = batch or make_batch()
    for name, value in vars(batch).items():
        if hasattr(state, name):
            setattr(state, name, value)
    state.input_ids = torch.zeros(batch.token_num, dtype=torch.int64)
    state.is_prefill = batch.phase is AttentionPhase.PREFILL
    state.att_state_manager = manager or AttentionStateManager(request_capacity=8)
    state.req_manager = SimpleNamespace(
        req_to_token_indexs=batch.req_to_token_indexs, HOLD_REQUEST_ID=batch.hold_req_id
    )
    state.microbatch_index = lane
    return state


def test_capture_manifest_records_owners_and_rejects_undeclared_model_tensor():
    state = make_state()
    backend = StubBackend(make_env(), make_spec())
    attention = coordinator([state], backend)
    weight = torch.ones(4)
    attention.binding = ModelBinding({"weights": SimpleNamespace(projection=weight)})
    with attention.session(state):
        state.attention.build_device_metadata()
        manifest = state.capture_manifest()
        entries = {entry.name: entry for entry in manifest.entries}
        assert entries["binding.weights.projection"].policy == "shared"
        assert entries["binding.weights.projection"].view.address == weight.data_ptr()
        assert entries["batch.req_to_token_indexs"].policy == "shared"
        assert entries["input_ids"].policy == "copy"
        state.new_model_rope_input = torch.zeros(2)
        with pytest.raises(ValueError, match="new_model_rope_input"):
            state.capture_manifest()


def test_manifest_checks_generated_and_host_prepared_addresses_before_replay():
    from lightllm.common.attention.api.graph import CaptureManifest

    generated = {"indices": torch.arange(4)}
    prepared = SimpleNamespace(indices=torch.arange(4), max_length=4)
    manifest = CaptureManifest()
    manifest.add("generated", "generated", generated, read=lambda: generated)
    manifest.add("planner.indices", "prepare", prepared.indices, read=lambda: prepared.indices)
    manifest.add("planner.length", "prepare", prepared.max_length, read=lambda: prepared.max_length)
    generated["indices"].fill_(3)
    prepared.indices.fill_(2)
    prepared.max_length = 2
    manifest.check()
    original = generated["indices"]
    generated["indices"] = original.clone()
    with pytest.raises(ValueError, match="generated.indices"):
        manifest.check()
    generated["indices"] = original
    prepared.indices = prepared.indices.clone()
    with pytest.raises(ValueError, match="planner.indices"):
        manifest.check(prepared=True)


def coordinator(states, backend):
    attention = ModelAttention.__new__(ModelAttention)
    attention.generation = states[0].att_state_manager.storage_generation
    attention.runtime = DeviceRuntime("cpu")
    attention.execution = ExecutionScope(attention.runtime)
    attention.groups = ({backend.phase: backend},)
    attention.workspaces = {}
    attention.replay_states = ()
    return attention


def test_submission_backpressure_retains_inputs_until_oldest_completion(monkeypatch):
    import gc
    import weakref

    runtime = DeviceRuntime("cpu")
    scope = ExecutionScope(runtime, max_pending=2)
    completed, events = [], []
    monkeypatch.setattr(runtime, "current_stream", lambda: "stream")

    def create_event():
        index = len(events)
        event = SimpleNamespace(
            record=lambda stream: None,
            query=lambda: index in completed,
            synchronize=lambda: completed.append(index),
        )
        events.append(event)
        return event

    monkeypatch.setattr(runtime, "create_event", create_event)
    source = torch.zeros(1)
    reference = weakref.ref(source)
    with scope.submit(retain=(source,)):
        pass
    del source
    with scope.submit():
        assert completed == []
        assert reference() is not None
    with scope.submit():
        gc.collect()
        assert completed == [0]
        assert reference() is None
    scope.synchronize()


def test_backpressure_failure_quarantines_without_releasing_inputs(monkeypatch):
    runtime = DeviceRuntime("cpu")
    scope = ExecutionScope(runtime, max_pending=1)
    source = object()
    monkeypatch.setattr(runtime, "current_stream", lambda: "stream")

    def fail():
        raise RuntimeError("device failed")

    monkeypatch.setattr(
        runtime,
        "create_event",
        lambda: SimpleNamespace(record=lambda stream: None, query=lambda: False, synchronize=fail),
    )
    with scope.submit(retain=(source,)):
        pass
    with pytest.raises(RuntimeError, match="device failed"):
        with scope.submit():
            pytest.fail("submitted after failed completion")
    assert scope._pending[0][1] == [source]
    with pytest.raises(RuntimeError, match="quarantined"):
        with scope.submit():
            pass


@pytest.mark.parametrize("invalid", ["conflicting", "partial", "source", "unchanged", "expanded"])
def test_replay_manifest_rejects_aliases_before_any_write(invalid):
    target = torch.zeros(8)
    source = torch.ones(8)
    copies = tensor_input_copy("first", target, source)
    if invalid == "conflicting":
        copies += tensor_input_copy("second", target, source.clone())
    elif invalid == "partial":
        copies += tensor_input_copy("second", target[1:], source[:-1])
    elif invalid == "source":
        copies += tensor_input_copy("second", torch.empty(8), target)
    elif invalid == "unchanged":
        copies += tensor_input_copy("second", target[:4], target[:4])
    else:
        copies = tensor_input_copy("expanded", target[:1].expand(8), source)
    with pytest.raises(ValueError, match="conflicting|overlap|overwrite"):
        validate_copy_plan(copies)
    assert target.eq(0).all()


def test_replay_manifest_coalesces_cumulative_offsets_and_prefix_views():
    target, source = torch.zeros(8), torch.arange(8, dtype=torch.float32)
    copies = tensor_input_copy("cumulative", target, source)
    copies += tensor_input_copy("start", target[:-1], source[:-1])
    copies += tensor_input_copy("duplicate", target, source)
    plan = validate_copy_plan(copies)
    assert len(plan) == 1
    for destination, incoming in plan:
        destination.copy_(incoming)
    torch.testing.assert_close(target, source)


def test_capture_survives_reuse_of_original_session(monkeypatch):
    backend = StubBackend(make_env(), make_spec())
    state = make_state()
    state.is_cuda_graph_capture = True
    attention = coordinator((state,), backend)
    plans = []
    monkeypatch.setattr(backend, "plan_host", lambda ctx: plans.append(ctx))
    with attention.session(state, max_kv_seq_len=16):
        batch = state.attention.batch
        context = state.attention.contexts[0]
        execution = CapturedGraph(
            SimpleNamespace(replay=lambda: None, reset=lambda: None), (state,), runtime=DeviceRuntime("cpu")
        )
    state.is_cuda_graph_capture = False
    state.b_seq_len = torch.tensor([9, 7], dtype=torch.int32)
    state.max_kv_seq_len = 9
    state.total_token_num = 16
    state.is_cuda_graph_replay = True
    with attention.session(state):
        plans.clear()
        execution.replay(state)
    assert execution.infer_states[0].attention.batch is batch
    assert execution.infer_states[0].attention is not state.attention
    assert len(plans) == 1 and plans[0] is context
    assert batch.b_seq_len.tolist() == [9, 7]
    assert batch.max_kv_seq_len == 16
    execution.close()


def test_rebinding_invalidates_capture_even_without_new_inputs():
    state = make_state()
    state.is_cuda_graph_capture = True
    backend = StubBackend(make_env(), make_spec())
    attention = coordinator((state,), backend)
    attention.binding = ModelBinding({"weights": object()})
    attention.routes = attention.adapters = ()
    graph = SimpleNamespace(replay=lambda: pytest.fail("submitted a stale binding"), reset=lambda: None)
    with attention.session(state):
        captured = CapturedGraph(graph, (state,), runtime=attention.runtime)
    old_binding = attention.binding
    attention.bind(
        SimpleNamespace(
            layers_infer=(),
            trans_layers_weight=(),
            pre_post_weight=object(),
            att_state_manager=state.att_state_manager,
        )
    )
    assert not old_binding.valid
    with pytest.raises(RuntimeError, match="model binding changed"):
        captured.replay()
    captured.close()


@pytest.mark.parametrize("failure", ["plan", "submit"])
def test_native_graph_failure_blocks_reuse_and_releases_graph(monkeypatch, failure):
    calls = []
    runtime = DeviceRuntime("cpu")
    monkeypatch.setattr(runtime, "synchronize", lambda: calls.append("drain"))

    def fail(*args):
        raise ValueError("native failure")

    backend = StubBackend(make_env(), make_spec())
    state = make_state()
    attention = coordinator((state,), backend)
    state.is_cuda_graph_capture = True
    with attention.session(state):
        execution = CapturedGraph(
            SimpleNamespace(replay=fail, reset=lambda: calls.append("reset")),
            (state,),
            runtime=runtime,
        )
    if failure == "plan":
        monkeypatch.setattr(backend, "plan_host", fail)
    with pytest.raises(ValueError, match="native") as raised:
        execution.replay(state)
    if failure == "submit":
        assert raised.value.attention_failure.stage == "graph submission"
        with pytest.raises(RuntimeError, match="worker recovery"):
            state.att_state_manager.check_execution_health()
    else:
        state.att_state_manager.check_execution_health()
    with pytest.raises(RuntimeError, match="invalid"):
        execution.replay(state)
    execution.close()
    assert calls == ["drain", "reset"]


@pytest.mark.parametrize("change", ["stride", "storage", "conflict"])
def test_compiled_manifest_revalidates_sources_and_destination_identity(change):
    target, source = torch.zeros(4, 4), torch.ones(4, 4)
    copies = [(target, source), (target, source)]
    plan = TensorCopyPlan([tensor for tensor, _ in copies])
    assert len(plan.validate(copies)) == 1
    if change == "stride":
        target.transpose_(0, 1)
    elif change == "storage":
        target.set_(torch.empty_like(target))
    else:
        copies[-1] = (target, source.clone())
    with pytest.raises(ValueError, match="destination"):
        plan.validate(copies)
