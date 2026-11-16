from unit_tests.common.attention.helpers import plan, run_attention


from dataclasses import replace


from types import SimpleNamespace


import pytest


import torch


from lightllm.common.attention.api import AttentionPhase


from unit_tests.common.attention.helpers import StubBackend, make_env, build


from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, make_spec


@pytest.mark.parametrize("capture", [False, True])
def test_context_capacity_and_backend_identity(capture):
    backend = StubBackend(make_env(), make_spec())
    context = make_context(backend, capture=capture, max_length=16)
    plan(backend, context, make_batch(req_to_token_indexs=context.batch.req_to_token_indexs))
    with pytest.raises(ValueError, match="capacity"):
        plan(backend, context, make_batch((17, 2)))
    with pytest.raises(ValueError, match="different attention backend"):
        plan(StubBackend(make_env(), make_spec()), context, make_batch())
    with pytest.raises(ValueError, match="different attention backend"):
        build(StubBackend(make_env(), make_spec()), context)
    attention = make_layer_attention(StubBackend(make_env(), make_spec()), context, params=None)
    with pytest.raises(ValueError, match="different attention backend"):
        run_attention(attention, context, inputs=None)


def test_layer_attention_rejects_wrong_phase():
    backend = StubBackend(make_env(), make_spec())
    context = make_context(backend, prepare=True)
    attention = make_layer_attention(backend, context, params=None)
    context.capacity = replace(context.capacity, phase=AttentionPhase.PREFILL)
    with pytest.raises(ValueError, match="capacity"):
        run_attention(attention, context, inputs=None)


class _OutputBackend(StubBackend):
    test_output_spec = staticmethod(lambda spec, inputs: None)

    def forward(self, layer, context, inputs, *, out=None):
        return inputs


@pytest.mark.parametrize("strided", [False, True])
def test_output_copy_and_padded_view_preserve_auxiliary_result(strided):
    backend = _OutputBackend(make_env(), make_spec())
    context = make_context(backend, prepare=True)
    layer = make_layer_attention(backend, context, params=None)
    out = torch.empty(2, 8 if strided else 4, 8)[:, :4]
    gate = torch.ones(2)
    output, actual_gate = run_attention(layer, context, (torch.ones(2, 32), gate), out=out)
    assert output is out and actual_gate is gate
    assert torch.all(out == 1)
    valid_heads = out[:, :2]
    output, actual_gate = run_attention(layer, context, (valid_heads, gate), out=out)
    assert output is valid_heads and actual_gate is gate


@pytest.mark.parametrize("invalid", ["dtype", "device", "size", "view_bounds"])
def test_output_rejects_incompatible_buffer(invalid):
    backend = _OutputBackend(make_env(), make_spec())
    context = make_context(backend, prepare=True)
    layer = make_layer_attention(backend, context, params=None)
    out = torch.empty(2, 4)
    result = torch.ones_like(out)
    if invalid == "dtype":
        result = result.double()
    elif invalid == "device":
        result = result.to("meta")
    elif invalid == "size":
        result = result[:1]
    else:
        result, out = out, out[:1]
    with pytest.raises(ValueError, match="out"):
        run_attention(layer, context, result, out=out)


def test_batch_builder_borrows_runtime_inputs_and_binding_checks_owner():
    from lightllm.common.basemodel.attention import build_attention_batch
    from lightllm.common.basemodel.infer_struct import InferStateInfo

    state = InferStateInfo()
    batch = make_batch()
    for name, value in vars(batch).items():
        if hasattr(state, name):
            setattr(state, name, value)
    state.input_ids = torch.zeros(batch.token_num, dtype=torch.long)
    state.is_prefill = False
    state.b_image_token_end = torch.tensor([1, 2])
    state.req_manager = SimpleNamespace(
        req_to_token_indexs=batch.req_to_token_indexs, HOLD_REQUEST_ID=batch.hold_req_id
    )
    borrowed = build_attention_batch(state)
    assert borrowed.b_req_idx is state.b_req_idx
    assert borrowed.b1_cu_q_seq_len is state.b1_cu_q_seq_len
    assert borrowed.req_to_token_indexs is state.req_manager.req_to_token_indexs
    assert borrowed.b_image_token_end is state.b_image_token_end
    backend = _OutputBackend(make_env(), make_spec())
    context = make_context(backend, borrowed, prepare=True)
    from lightllm.common.attention.api import AttentionWorkspace

    layer = make_layer_attention(backend, context, params=None)
    with state.attention.prepare(
        borrowed, ({borrowed.phase: backend},), context.capacity, workspace=lambda _: AttentionWorkspace("cpu")
    ):
        context = state.attention.contexts[0]
        assert layer.context(state.attention) is context
        output = torch.ones(2, 32)
        with state.attention.layer_pass():
            assert layer.forward(state.attention, output) is output
        with pytest.raises(ValueError, match="different attention backend"):
            replace(layer, backend=_OutputBackend(make_env(), make_spec())).forward(state.attention, output)
        with pytest.raises(ValueError, match="different attention backend"):
            replace(layer, backend=StubBackend(make_env(), make_spec())).context(state.attention)
    with pytest.raises(AttributeError):
        context.b_requ_idx = borrowed.b_req_idx


def test_binding_resolves_context_on_each_forward(monkeypatch):
    backend = _OutputBackend(make_env(), make_spec())
    first, second = make_context(backend), make_context(backend)
    layer = make_layer_attention(backend, first, params=None)
    forwarded = []

    def forward(layer, context, inputs, *, out=None):
        forwarded.append(context)
        return inputs

    monkeypatch.setattr(backend, "forward", forward)
    result, out = torch.ones(2, 32), torch.empty(2, 32)
    for context in (first, second):
        assert run_attention(layer, context, result, out=out) is out
    assert all(a is b for a, b in zip(forwarded, (first, second)))
    torch.testing.assert_close(out, result)


def test_session_enforces_direct_backend_contract():
    from lightllm.common.attention.api import AttentionBackend, AttentionContext, AttentionWorkspace, ContextCapacity
    from lightllm.common.attention.session import AttentionSession

    class Backend:
        phase = AttentionPhase.DECODE
        spec = make_spec()

        @classmethod
        def check_environment(cls, caps):
            pass

        @classmethod
        def check_support(cls, spec, caps=None, constraints=None):
            pass

        def allocate_context(self, capacity, *, capture, workspace):
            return AttentionContext(capacity, capture, workspace)

        def resource_requirements(self, capacity, *, capture):
            from lightllm.common.attention.api import WorkspaceBudget

            return WorkspaceBudget()

        def plan_host(self, context):
            pass

        def build_device_metadata(self, context):
            pass

        def forward(self, layer, context, inputs, *, out=None):
            return inputs

    backend, session = Backend(), AttentionSession()
    batch = make_batch()
    capacity = ContextCapacity(phase=batch.phase, batch_size=2, token_num=2, max_kv_seq_len=16, max_q_seq_len=1)
    with session.prepare(
        batch, ({batch.phase: backend},), capacity, workspace=lambda _: AttentionWorkspace("cpu"), capture=True
    ):
        context = session.contexts[0]
        layer = make_layer_attention(StubBackend(make_env(), make_spec()), context, None)
        layer = replace(layer, backend=backend, output_spec=lambda spec, inputs: None)
        out = torch.zeros(2, 4)
        with session.layer_pass():
            assert layer.forward(session, torch.ones(2, 4), out=out) is out
        assert out.eq(1).all()
        with pytest.raises(ValueError, match="different attention backend"):
            replace(layer, backend=Backend()).forward(session, out)
        context.capacity = replace(context.capacity, phase=AttentionPhase.PREFILL)
        with pytest.raises(ValueError, match="different attention backend"):
            layer.forward(session, out)


@pytest.mark.parametrize("failure", ["allocate", "plan", "forward"])
def test_session_releases_workspace_on_error(failure):
    from lightllm.common.attention.api import AttentionWorkspace, ContextCapacity
    from lightllm.common.attention.session import AttentionSession
    from lightllm.platform import DeviceRuntime

    events = []

    from lightllm.common.execution import ExecutionScope

    class Scope(ExecutionScope):
        def _begin(self):
            events.append("begin")

        def _finish(self, submission):
            events.append("finish")

    class Backend(StubBackend):
        def allocate_context(self, *args, **kwargs):
            if failure == "allocate":
                raise RuntimeError("allocate failed")
            return super().allocate_context(*args, **kwargs)

        def plan_host(self, context):
            if failure == "plan":
                raise RuntimeError("plan failed")

    session = AttentionSession()
    batch = make_batch()
    backend = Backend(make_env(), make_spec())
    runtime = DeviceRuntime("cpu")
    workspace = AttentionWorkspace(runtime=runtime, execution=Scope(runtime))
    capacity = ContextCapacity(phase=batch.phase, batch_size=2, token_num=2, max_kv_seq_len=16, max_q_seq_len=1)
    with pytest.raises(RuntimeError, match=failure):
        with session.prepare(batch, ({batch.phase: backend},), capacity, workspace=lambda _: workspace):
            raise RuntimeError("forward failed")
    assert events == ["begin", "finish"]


def test_model_capture_session_uses_graph_bound_and_one_batch():
    from lightllm.common.attention.api import AttentionWorkspace
    from lightllm.common.attention.session import AttentionSession
    from lightllm.common.basemodel.attention import ModelAttention
    from lightllm.platform import DeviceRuntime

    events = []

    from lightllm.common.execution import ExecutionScope

    class Scope(ExecutionScope):
        def _begin(self):
            events.append("begin")

        def _finish(self, submission):
            events.append("finish")

    attention = ModelAttention.__new__(ModelAttention)
    attention.runtime = DeviceRuntime("cpu")
    attention.execution = Scope(attention.runtime)
    attention.generation = 3
    attention.constraints = SimpleNamespace(max_seq_len=128)
    attention.groups = tuple({AttentionPhase.DECODE: StubBackend(make_env(), make_spec())} for _ in range(2))
    attention.workspaces = {}
    batch = make_batch((9, 7))
    state = SimpleNamespace(
        **vars(batch),
        is_prefill=False,
        input_ids=torch.zeros(batch.token_num, dtype=torch.int64),
        req_manager=SimpleNamespace(req_to_token_indexs=batch.req_to_token_indexs, HOLD_REQUEST_ID=batch.hold_req_id),
        att_state_manager=SimpleNamespace(storage_generation=3, check_execution_health=lambda: None),
        attention=AttentionSession(),
        microbatch_index=0,
        is_cuda_graph_capture=True,
        is_cuda_graph_replay=False,
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        with attention.session(state, max_kv_seq_len=16) as session:
            assert session.batch.max_kv_seq_len == 16
            assert session.batch.total_token_num == 32
            for context in session.contexts.values():
                assert context.batch is session.batch
                assert context.capture
                assert context.capacity.max_kv_seq_len == 16
                assert context.current_max_kv_seq_len == 9
            assert len(session.contexts) == 2
            assert state.max_kv_seq_len == 9
            raise RuntimeError("forward failed")

    assert events == ["begin", "finish"]


@pytest.mark.parametrize("capture", [False, True])
def test_session_execution_ends_with_scope(capture):
    from lightllm.common.attention.api import AttentionWorkspace, ContextCapacity, LayerAttention
    from lightllm.common.attention.session import AttentionSession

    session = AttentionSession()
    backend = _OutputBackend(make_env(), make_spec())
    batch = make_batch()
    capacity = ContextCapacity(phase=batch.phase, batch_size=2, token_num=2, max_kv_seq_len=16, max_q_seq_len=1)
    layer = LayerAttention(0, 0, None, backend, backend.test_output_spec)
    value = torch.ones(2)
    with session.prepare(
        batch, ({batch.phase: backend},), capacity, workspace=lambda _: AttentionWorkspace("cpu"), capture=capture
    ):
        with pytest.raises(RuntimeError, match="active layer pass"):
            layer.forward(session, value)
        with session.layer_pass():
            assert layer.forward(session, value) is value
        with pytest.raises(RuntimeError, match="already active"):
            with session.prepare(batch, (), capacity, workspace=lambda _: None):
                pass
    assert session.contexts[0].batch is not None
    for operation in (lambda: layer.forward(session, value), session.build_device_metadata):
        with pytest.raises(RuntimeError, match="active session"):
            operation()


@pytest.mark.parametrize("changed", ["capacity", "workspace", "capture"])
def test_session_rejects_context_substitution(changed):
    from lightllm.common.attention.api import AttentionContext, AttentionWorkspace, ContextCapacity
    from lightllm.common.attention.session import AttentionSession

    class Backend(StubBackend):
        def allocate_context(self, capacity, *, capture, workspace):
            if changed == "capacity":
                capacity = replace(capacity, max_kv_seq_len=32)
            elif changed == "workspace":
                workspace = AttentionWorkspace("cpu")
            else:
                capture = not capture
            return AttentionContext(capacity, capture, workspace)

    batch = make_batch()
    backend = Backend(make_env(), make_spec())
    capacity = ContextCapacity(phase=batch.phase, batch_size=2, token_num=2, max_kv_seq_len=16, max_q_seq_len=1)
    with pytest.raises(ValueError, match="backend changed"):
        with AttentionSession().prepare(
            batch, ({batch.phase: backend},), capacity, workspace=lambda _: AttentionWorkspace("cpu")
        ):
            pytest.fail("invalid context entered execution")


def test_output_rejects_strided_padding_hole():
    backend = _OutputBackend(make_env(), make_spec())
    context = make_context(backend, prepare=True)
    layer = make_layer_attention(backend, context, None)
    base = torch.arange(16).view(2, 8)
    with pytest.raises(ValueError, match="out"):
        run_attention(layer, context, base[:1, 4:8], out=base[:, :4])
