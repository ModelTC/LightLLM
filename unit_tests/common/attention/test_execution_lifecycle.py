from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase, ContextCapacity, LayerAttention
from lightllm.common.attention.api.graph import TensorCopyPlan, tensor_input_copy, validate_copy_plan
from lightllm.common.attention.api.inputs import DenseInputs, DenseResources
from lightllm.common.attention.api.workspace import AttentionWorkspace
from lightllm.common.attention.session import AttentionSession
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.basemodel.attention import ModelAttention, ModelBinding, describe_serving_constraints
from lightllm.common.basemodel.attention_adapters import DenseAdapter
from lightllm.common.basemodel.graph import CapturedGraph
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.execution import ExecutionScope
from lightllm.platform import DeviceRuntime
from unit_tests.common.attention.helpers import StubBackend, make_batch, make_env, make_spec, model_stub


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


def preparation_model(runtime, prepare):
    from contextlib import contextmanager
    from lightllm.common.basemodel.basemodel import TpPartBaseModel

    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.device_runtime = runtime
    model.att_state_manager = AttentionStateManager(request_capacity=2)
    model.args = SimpleNamespace(enable_prefill_decode_mixed=False, enable_tpsp_mix_mode=False, mtp_step=0)
    model.prefill_graph = None
    model._create_padded_prefill_model_input = lambda **kwargs: kwargs["model_input"]
    model._create_inferstate = lambda _: SimpleNamespace(init_some_extra_state=prepare)
    model._context_forward = lambda **kwargs: SimpleNamespace()
    model._create_unpad_prefill_model_output = lambda **kwargs: kwargs["padded_model_output"]

    @contextmanager
    def session(state):
        with model.execution.join():
            yield

    model.attention = SimpleNamespace(session=session)
    return model


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


def test_submission_fences_before_model_preparation_and_records_one_completion(monkeypatch):
    calls = []
    runtime = DeviceRuntime("cpu")
    stream = SimpleNamespace(wait_event=lambda event: calls.append("fence"))
    monkeypatch.setattr(runtime, "current_stream", lambda: stream)
    monkeypatch.setattr(
        runtime, "create_event", lambda: SimpleNamespace(record=lambda stream: calls.append("completion"))
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None))
    model = preparation_model(runtime, lambda model: calls.append("prepare"))
    model.execution.record_reader(object())
    model.forward(
        SimpleNamespace(is_prefill=True, input_ids=torch.ones(2), batch_size=1, to_cuda=lambda: calls.append("input"))
    )
    assert calls == ["fence", "input", "prepare", "completion"]


@pytest.mark.parametrize("entry", ["forward", "microbatch_overlap_prefill", "microbatch_overlap_decode"])
def test_public_submission_rejects_overlap_before_touching_inputs(entry):
    model = preparation_model(DeviceRuntime("cpu"), lambda model: None)
    incoming = SimpleNamespace(to_cuda=lambda: pytest.fail("input touched without ownership"))
    with model.execution.submit():
        with pytest.raises(RuntimeError, match="active submission"):
            getattr(model, entry)(*([incoming] if entry == "forward" else [incoming, incoming]))
    model.att_state_manager.check_execution_health()


@pytest.mark.parametrize("forward_error", [False, True])
def test_model_completion_failure_blocks_state_even_after_session_exit(monkeypatch, forward_error):
    runtime = DeviceRuntime("cpu")
    model = preparation_model(runtime, lambda model: None)
    state = make_state()
    model.att_state_manager = state.att_state_manager
    monkeypatch.setattr(runtime, "current_stream", lambda: "stream")
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None))

    def fail_completion():
        raise RuntimeError("completion failed")

    def forward(**kwargs):
        model.execution.retain(state)
        state.attention.state_may_be_modified = True
        if forward_error:
            raise ValueError("forward failed")
        return SimpleNamespace()

    monkeypatch.setattr(runtime, "create_event", fail_completion)
    model._context_forward = forward
    incoming = SimpleNamespace(is_prefill=True, input_ids=torch.ones(2), batch_size=1, to_cuda=lambda: None)
    expected = ValueError if forward_error else RuntimeError
    with pytest.raises(expected, match="forward failed" if forward_error else "completion failed"):
        model.forward(incoming)
    with pytest.raises(RuntimeError, match="worker recovery"):
        model.att_state_manager.check_execution_health()
    with pytest.raises(RuntimeError, match="quarantined"):
        with model.execution.submit():
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_preparation_shared_buffer_waits_for_previous_model_stream():
    from lightllm.platform import get_device_runtime

    buffer = torch.zeros(4096, dtype=torch.int32, device="cuda")
    observations = []

    def prepare(model):
        if not observations:
            torch.cuda._sleep(2_000_000)
            buffer.fill_(17)
        observations.append(buffer.clone())

    model = preparation_model(get_device_runtime("cuda"), prepare)
    incoming = SimpleNamespace(is_prefill=True, input_ids=torch.ones(2), batch_size=1, to_cuda=lambda: None)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        model.forward(incoming)
    with torch.cuda.stream(consumer):
        model.forward(incoming)
    consumer.synchronize()
    assert observations[1].eq(17).all()


def test_model_scope_rejects_competing_lanes_before_preparation(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    first = make_state()
    second = make_state(manager=first.att_state_manager, lane=1)
    backend = StubBackend(make_env(), make_spec())
    attention = coordinator((first, second), backend)
    first.attention.state_may_be_modified = True
    monkeypatch.setattr(backend, "plan_host", lambda ctx: pytest.fail("prepared before all lanes were acquired"))

    def compete():
        with pytest.raises(RuntimeError, match="active submission"):
            with attention.session(first, second):
                pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        with attention.execution.submit():
            executor.submit(compete).result(timeout=5)
    with attention.execution.submit():
        pass
    first.att_state_manager.check_execution_health()


@pytest.mark.parametrize("failure", ["layer", "epilogue", "head", "completion"])
def test_partial_forward_blocks_continuation_and_publication(monkeypatch, failure):
    state = make_state()
    backend = StubBackend(make_env(), make_spec())
    attention = coordinator((state,), backend)

    def fail(*args):
        raise RuntimeError(failure)

    if failure == "epilogue":
        monkeypatch.setattr(attention, "finish_decode", fail)
    if failure == "completion":
        monkeypatch.setattr(attention.runtime, "current_stream", lambda: "stream")
        monkeypatch.setattr(attention.runtime, "create_event", fail)
    with pytest.raises(RuntimeError, match=failure) as raised:
        with attention.session(state):
            with attention.layer_pass(state):
                if failure == "layer":
                    fail()
            if failure == "head":
                fail()
    assert raised.value.attention_failure.role == "main"
    assert raised.value.attention_failure.request_indices[0] is state.b_req_idx
    manager = state.att_state_manager
    with pytest.raises(RuntimeError, match="worker recovery"):
        with attention.session(make_state(manager=manager)):
            pytest.fail("reused partial state")
    for publish in (
        lambda: manager.save_bounded_att_runtime_to_pages([], [], "big", []),
        lambda: manager.commit_bounded_att_state(None, None, None, "main"),
        lambda: manager.get_cpu_cache_operator().write_req(None, None, [], {}),
        lambda: manager.get_pd_operator().write_page(None, 0, "kv", {}),
    ):
        with pytest.raises(RuntimeError, match="worker recovery"):
            publish()


def test_each_callback_has_one_metadata_build_and_epilogue(monkeypatch):
    state = make_state()
    backend = StubBackend(make_env(), make_spec())
    attention = coordinator((state,), backend)
    calls = []
    monkeypatch.setattr(backend, "build_device_metadata", lambda ctx: calls.append("metadata"))
    monkeypatch.setattr(attention, "finish_decode", lambda state: calls.append("finish"))
    with attention.session(state):
        for _ in range(3):
            with attention.layer_pass(state):
                calls.append("layers")
                with pytest.raises(RuntimeError, match="already active"):
                    with state.attention.layer_pass():
                        pass
    assert calls == ["metadata", "layers", "finish"] * 3


@pytest.mark.parametrize("invalid", ["dtype", "shape", "query_alias", "state_alias", "overlapping_output"])
def test_output_preflight_prevents_backend_mutation(monkeypatch, invalid):
    state = make_state()
    backend = StubBackend(make_env(), make_spec())
    attention = coordinator((state,), backend)
    query, key = torch.ones(2, 4, 8), torch.zeros(2, 4, 8)
    layer = LayerAttention(0, 0, DenseResources(key, key.clone()), backend, DenseAdapter.output_spec)
    out = torch.empty_like(query)
    if invalid == "dtype":
        out = out.double()
    elif invalid == "shape":
        out = out[:, :, :4]
    elif invalid == "query_alias":
        out = query
    elif invalid == "state_alias":
        out = key
    else:
        out = torch.zeros(1, 4, 8).expand(2, 4, 8)
    monkeypatch.setattr(backend, "forward", lambda *args, **kwargs: pytest.fail("invalid output reached backend"))
    with attention.session(state):
        with attention.layer_pass(state):
            with pytest.raises(ValueError, match="out"):
                layer.forward(state.attention, DenseInputs(query), out=out)
    assert key.eq(0).all()


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


@pytest.mark.parametrize("failing", [False, True])
def test_weight_reload_owns_submission_and_drains_before_loading(monkeypatch, failing):
    from lightllm.common.basemodel import basemodel

    calls = []
    model = basemodel.TpPartBaseModel.__new__(basemodel.TpPartBaseModel)
    model.device_runtime = DeviceRuntime("cpu")
    model.att_state_manager = AttentionStateManager(request_capacity=2)
    binding = ModelBinding({})
    model.attention = SimpleNamespace(generation=0, binding=binding, bind=lambda _: calls.append("bind"))
    model.data_type, model.weight_dir_ = torch.float32, None
    model.pre_post_weight = SimpleNamespace(verify_load=lambda: calls.append("verify"))
    model.trans_layers_weight = []
    captured = CapturedGraph(
        SimpleNamespace(reset=lambda: calls.append("reset")),
        runtime=model.device_runtime,
        execution=model.execution,
    )
    model.att_state_manager.register_graph(captured)
    monkeypatch.setattr(model.device_runtime, "synchronize", lambda: calls.append("drain"))

    def load(**kwargs):
        assert not binding.valid
        assert captured.graph is None
        calls.append("load")
        if failing:
            raise ValueError("bad weight")

    monkeypatch.setattr(basemodel, "load_hf_weights", load)
    with model.execution.submit():
        with pytest.raises(RuntimeError, match="active submission"):
            model.load_weights({})
    assert not calls
    if failing:
        with pytest.raises(ValueError, match="bad weight"):
            model.load_weights({})
        with pytest.raises(RuntimeError, match="worker recovery"):
            model.att_state_manager.check_execution_health()
    else:
        model.load_weights({})
        assert calls[-2:] == ["verify", "bind"]
    assert calls[:4] == ["drain", "drain", "reset", "load"]


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


def test_phase_reservation_bounds_and_unsupported_overlap():
    model = model_stub(disable_cudagraph=False)
    model.max_req_num = 12
    model.batch_max_tokens = 128
    model.graph_max_batch_size = 24
    model.graph_max_len_in_batch = 512
    model.mtp_manager.get_decode_tokens_per_request = lambda _: 3
    model._align_decode_batch_size = lambda size: ((size + 7) // 8) * 8
    attention = ModelAttention.__new__(ModelAttention)
    attention.constraints = describe_serving_constraints(model)
    prefill = attention.reservation_capacity(model, AttentionPhase.PREFILL, capture=False)
    decode = attention.reservation_capacity(model, AttentionPhase.DECODE, capture=False)
    capture = attention.reservation_capacity(model, AttentionPhase.DECODE, capture=True)
    assert (prefill.batch_size, prefill.token_num, prefill.max_q_seq_len) == (13, 128, 128)
    assert (decode.batch_size, decode.token_num, decode.max_q_seq_len) == (40, 40, 3)
    assert (capture.token_num, capture.max_kv_seq_len) == (24, 512)
    model.args.enable_prefill_cudagraph = model.args.enable_prefill_microbatch_overlap = True
    with pytest.raises(ValueError, match="does not support microbatch overlap"):
        describe_serving_constraints(model)
