from unit_tests.common.attention.helpers import plan, run_attention
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import registry, selection
from lightllm.common.attention.api import AttentionFamily, AttentionPhase, UnsupportedAttention
from unit_tests.common.attention.helpers import StubBackend, make_env, phase_class, build
from lightllm.utils.backend_validator import ProbeResult, ProbeScope
from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, make_spec, model_stub


def _select(*, model, specs=None, **kwargs):
    family, phase = kwargs.pop("family"), kwargs.pop("phase")
    candidates = kwargs.pop("candidates", None)
    priority = kwargs.pop("priority", ())
    if candidates is not None:
        kwargs["policy"] = selection.BackendPolicy(
            allow=frozenset(priority),
            prefer=tuple(priority),
            extra=tuple(backend for formats in candidates.values() for backend in formats.values()),
        )
    return selection.select_attention_backend(
        specs=specs or (make_spec(family, phase),),
        caps=model.device_runtime.caps,
        execution=model.attention_requirements,
        **kwargs,
    )


def test_graph_copy_validates_all_inputs_before_writing():
    captured = make_batch()
    incoming = make_batch((9, 7))
    incoming = replace(incoming, req_to_token_indexs=captured.req_to_token_indexs, b_seq_len=incoming.b_seq_len.long())
    before = captured.b_req_idx.clone()
    with pytest.raises(ValueError, match="b_seq_len"):
        captured.copy_inputs(incoming)
    torch.testing.assert_close(captured.b_req_idx, before)


def test_graph_copy_keeps_tensor_addresses_and_uses_new_lengths():
    captured = make_batch()
    incoming = make_batch((9, 7), req_to_token_indexs=captured.req_to_token_indexs)
    pointer = captured.b_seq_len.data_ptr()
    captured.copy_inputs(incoming)
    assert captured.b_seq_len.data_ptr() == pointer
    assert captured.b_seq_len.tolist() == [9, 7]
    with pytest.raises(ValueError, match="request table"):
        captured.copy_inputs(make_batch((9, 7)))


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
    with pytest.raises(ValueError, match="different attention backend"):
        run_attention(attention, context, inputs=None)


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_selection_never_falls_back_to_a_rejected_backend(monkeypatch, requested):
    spec = make_spec(cache=replace(make_spec().cache, encoding="fp8kv_sph"))
    monkeypatch.setattr(
        StubBackend,
        "check_environment",
        classmethod(lambda cls, caps: None),
    )
    with pytest.raises(selection.BackendSelectionError, match="encoding fp8kv_sph"):
        _select(
            family=spec.family,
            phase=spec.phase,
            model=model_stub(),
            specs=(spec,),
            requested=requested,
            candidates={"None": {"triton": StubBackend}},
            priority=["triton"],
        )


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_environment_checks_apply_to_explicit_and_auto(monkeypatch, requested):
    def reject(cls, caps):
        raise UnsupportedAttention("missing extension")

    monkeypatch.setattr(StubBackend, "check_environment", classmethod(reject))
    with pytest.raises(selection.BackendSelectionError, match="missing extension"):
        _select(
            family=AttentionFamily.DENSE,
            phase=AttentionPhase.DECODE,
            model=model_stub(),
            requested=requested,
            candidates={"None": {"triton": StubBackend}},
            priority=["triton"],
        )


def _backend_check_exception_candidates(monkeypatch, error_type=RuntimeError):
    class BrokenBackend(StubBackend):
        provider = "broken"

        @classmethod
        def check_environment(cls, caps):
            raise error_type("provider ABI mismatch")

    class WorkingBackend(StubBackend):
        provider = "working"

        @classmethod
        def check_environment(cls, caps):
            return None

    monkeypatch.setattr(
        "lightllm.utils.backend_validator.probe_backend",
        lambda *args, **kwargs: ProbeResult(True, ProbeScope.BACKEND_NUMERICS),
    )
    return {"None": {"broken": BrokenBackend, "working": WorkingBackend}}


@pytest.mark.parametrize("error_type", [UnsupportedAttention, ValueError, RuntimeError])
def test_backend_check_exception_allows_auto_fallback(monkeypatch, error_type):
    candidates = _backend_check_exception_candidates(monkeypatch, error_type)
    selected = _select(
        family=AttentionFamily.DENSE,
        phase=AttentionPhase.DECODE,
        model=model_stub(),
        requested="auto",
        candidates=candidates,
        priority=["broken", "working"],
    )
    assert selected.provider == "working"


@pytest.mark.parametrize("error_type", [UnsupportedAttention, ValueError, RuntimeError])
def test_backend_check_exception_is_reported_for_explicit_selection(monkeypatch, error_type):
    candidates = _backend_check_exception_candidates(monkeypatch, error_type)
    with pytest.raises(selection.BackendSelectionError) as error:
        _select(
            family=AttentionFamily.DENSE,
            phase=AttentionPhase.DECODE,
            model=model_stub(),
            requested="broken",
            candidates=candidates,
            priority=["broken", "working"],
        )

    reason = "provider ABI mismatch"
    if error_type is not UnsupportedAttention:
        reason = f"provider check failed: {error_type.__name__}: {reason}"
    assert error.value.failures == {"broken": reason}


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_rejection_reaches_rank_consensus(monkeypatch, requested):
    spec = make_spec(dtype=torch.float64)
    ranks = []
    monkeypatch.setattr(StubBackend, "check_environment", classmethod(lambda cls, caps: None))
    monkeypatch.setattr(
        selection.backend_validator, "probe_backend", lambda *args, **kwargs: pytest.fail("probed rejected inputs")
    )
    monkeypatch.setattr(selection, "_check_rank_selection", lambda *args: ranks.append(args))
    with pytest.raises(selection.BackendSelectionError, match="activation dtype"):
        _select(
            family=spec.family,
            phase=spec.phase,
            model=model_stub(),
            specs=(spec,),
            requested=requested,
            candidates={"None": {"triton": StubBackend}},
            priority=["triton"],
        )
    assert len(ranks) == 1
    assert ranks[0][:3] == (spec.family, spec.phase, None)
    assert "activation dtype" in ranks[0][3]


def test_rank_selection_checks_all_ranks_after_local_selection(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather(output, value):
        output[:] = [value, {"backend": "triton", "failure": None}]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(selection.BackendSelectionError, match="backend selection"):
        selection._check_rank_selection(AttentionFamily.DENSE, AttentionPhase.DECODE, "fa3", None)


def test_rank_selection_propagates_remote_failure(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, {"backend": None, "failure": "probe failed"}]),
    )
    with pytest.raises(selection.BackendSelectionError, match="rank 1: probe failed"):
        selection._check_rank_selection(AttentionFamily.DENSE, AttentionPhase.DECODE, "fa3", None)


@pytest.mark.parametrize("format,provider", [("fp8kv_sph", "flashinfer"), ("fp8kv_spt", "fa3")])
def test_fp8_scale_layouts_are_not_interchangeable(format, provider):
    from lightllm.common.attention.api.spec import CacheEncoding

    backends = (registry.load_backend_class(ref) for ref in registry.BACKENDS)
    assert not any(cls.provider == provider and CacheEncoding(format) in cls.cache_encodings for cls in backends)


def test_layer_shape_configures_group_planner():
    from lightllm.common.attention.backends.dense.flashinfer.backend import FlashInferDenseDecodeBackend

    model = model_stub(page_size=16)
    model.config = {"num_attention_heads": 16, "num_key_value_heads": 4, "hidden_size": 2048}
    model.max_seq_length = 128
    model.data_type = torch.bfloat16
    small = FlashInferDenseDecodeBackend(make_env(model), make_spec(num_q_heads=8, head_size=64))
    large = FlashInferDenseDecodeBackend(make_env(model), make_spec(num_q_heads=16, head_size=128))
    assert (small.tp_q_head_num, small.head_dim) == (8, 64)
    assert (large.tp_q_head_num, large.head_dim) == (16, 128)


def test_groups_follow_each_layers_family_and_shape(monkeypatch):
    from lightllm.common.basemodel.attention import ModelAttention
    from lightllm.common.basemodel.attention_adapters import ADAPTERS, AttentionDescription

    specs = [make_spec(head_size=64), make_spec(head_size=64), make_spec(head_size=128), make_spec(AttentionFamily.GDN)]
    calls = []

    def select(**kwargs):
        calls.append(kwargs)
        return lambda env, spec: SimpleNamespace(spec=spec, family=spec.family)

    monkeypatch.setattr(selection, "select_attention_backend", select)
    for adapter in ADAPTERS.values():
        monkeypatch.setattr(adapter, "describe", lambda model, layer, phase, desc: replace(layer.spec, phase=phase))
    model = model_stub()
    model.attention_policy = lambda family, phase: selection.BackendPolicy()
    model.layers_infer = [
        SimpleNamespace(
            layer_num_=i,
            spec=spec,
            describe_attention=lambda phase, s=spec: AttentionDescription(s.family),
            attention_adapter=lambda family: ADAPTERS[family],
        )
        for i, spec in enumerate(specs)
    ]
    attention = ModelAttention(model)
    assert attention.routes == (0, 0, 1, 2)
    assert len(calls) == 6
    assert [group[AttentionPhase.PREFILL].spec.head_size for group in attention.groups[:2]] == [64, 128]
    assert attention.groups[-1][AttentionPhase.DECODE].family is AttentionFamily.GDN


@pytest.mark.parametrize("invalid", ["input_ids", "group", "role"])
def test_runner_graph_copy_rejects_before_mutating_batch(invalid):
    from lightllm.common.basemodel.infer_struct import InferStateInfo

    captured, incoming = InferStateInfo(), InferStateInfo()
    captured.att_state_role = incoming.att_state_role = "main"
    group = object()
    captured.attention.groups = incoming.attention.groups = (group,)
    captured.attention.batch = make_batch()
    incoming.attention.batch = make_batch((9, 7), req_to_token_indexs=captured.attention.batch.req_to_token_indexs)
    captured.input_ids = torch.zeros(2, dtype=torch.long)
    incoming.input_ids = torch.ones(3 if invalid == "input_ids" else 2, dtype=torch.long)
    if invalid == "group":
        incoming.attention.groups = (object(),)
    if invalid == "role":
        incoming.att_state_role = "draft"
    before = captured.attention.batch.b_seq_len.clone()
    with pytest.raises(ValueError):
        captured.graph_input_copies(incoming)
    torch.testing.assert_close(captured.attention.batch.b_seq_len, before)
    assert captured.input_ids.tolist() == [0, 0]


def test_replay_flashqla_rejects_decode_and_fp32():
    from lightllm.common.attention.backends.gdn.replay.backend import FlashQlaGDNReplayPrefillBackend

    spec = make_spec(family=AttentionFamily.GDN)
    spec = replace(spec, cache=replace(spec.cache, state_mode="replay"))
    with pytest.raises(UnsupportedAttention, match="decode"):
        FlashQlaGDNReplayPrefillBackend.check_support(spec)
    with pytest.raises(UnsupportedAttention, match="dtype"):
        FlashQlaGDNReplayPrefillBackend.check_support(replace(spec, phase=AttentionPhase.PREFILL, dtype=torch.float32))


class _OutputBackend(StubBackend):
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
    state.attention.contexts = {0: context}
    layer = make_layer_attention(backend, context, params=None)
    assert layer.context(state.attention) is context
    output = torch.ones(2, 32)
    assert layer.forward(state.attention, output) is output
    with pytest.raises(ValueError, match="different attention backend|specifications differ"):
        replace(layer, backend=_OutputBackend(make_env(), make_spec())).forward(state.attention, output)
    with pytest.raises(ValueError, match="different attention backend|specifications differ"):
        replace(layer, backend=StubBackend(make_env(), make_spec())).context(state.attention)
    with pytest.raises(AttributeError):
        context.b_requ_idx = borrowed.b_req_idx


@pytest.mark.parametrize("family", list(AttentionFamily))
def test_probe_reports_actual_coverage(monkeypatch, family):
    from lightllm.common.attention import probe, registry
    from lightllm.utils import backend_validator as validator

    spec = make_spec(family)
    calls = []
    monkeypatch.setattr(registry, "load_backend_class", lambda name: SimpleNamespace(provider="triton"))
    monkeypatch.setattr(probe, "probe_dense", lambda *args: calls.append("backend") or (True, None))
    monkeypatch.setattr(validator, "_smoke_provider", lambda *a, **k: calls.append("smoke") or (True, None))
    result = validator._probe_in_process("selected", device="cpu", spec=spec)
    assert result.ok
    numerical = family is AttentionFamily.DENSE
    assert result.scope is (validator.ProbeScope.BACKEND_NUMERICS if numerical else validator.ProbeScope.PROVIDER_SMOKE)
    assert calls == (["backend"] if numerical else ["smoke"])


def test_special_dense_probe_is_explicitly_smoke_only(monkeypatch):
    from dataclasses import replace

    from lightllm.common.attention import probe, registry
    from lightllm.utils import backend_validator as validator

    monkeypatch.setattr(registry, "load_backend_class", lambda name: SimpleNamespace(provider="triton"))
    monkeypatch.setattr(probe, "probe_dense", lambda *args: pytest.fail("unsupported numerical probe"))
    monkeypatch.setattr(validator, "_smoke_provider", lambda *a, **k: (False, "provider failed"))
    result = validator._probe_in_process("selected", device="cpu", spec=make_spec(use_mtp=True))
    assert not result.ok
    assert result.scope is validator.ProbeScope.PROVIDER_SMOKE
    assert result.reason == "provider failed"


def test_replay_flashqla_uses_inherited_initialization(monkeypatch):
    import sys
    from types import ModuleType

    from lightllm.common.attention.backends.gdn.replay.backend import (
        FlashQlaGDNReplayPrefillBackend,
        TritonGDNReplayDecodeBackend,
    )

    calls = []
    original = phase_class(TritonGDNReplayDecodeBackend, AttentionPhase.PREFILL).__init__

    def initialize(self, model, spec):
        calls.append("parent")
        original(self, model, spec)

    monkeypatch.setattr(phase_class(TritonGDNReplayDecodeBackend, AttentionPhase.PREFILL), "__init__", initialize)
    monkeypatch.setattr(
        phase_class(TritonGDNReplayDecodeBackend, AttentionPhase.PREFILL),
        "_init_linear_layer_metadata",
        lambda *a: calls.append("metadata"),
    )
    module = ModuleType("flash_qla")
    module.chunk_gated_delta_rule = object()
    monkeypatch.setitem(sys.modules, "flash_qla", module)
    model = model_stub()
    backend = FlashQlaGDNReplayPrefillBackend(make_env(model), make_spec(AttentionFamily.GDN))
    assert calls == ["parent", "metadata"]
    assert backend.runtime is model.device_runtime
    assert backend.prefill_kernel is module.chunk_gated_delta_rule


def test_spec_configures_backend_before_planning():
    from lightllm.common.attention.api import AttentionContext, AttentionWorkspace

    model = model_stub(page_size=1, mtp_dynamic_verify=False)
    spec = make_spec(cache=replace(make_spec().cache, page_size=16), causal=False, dynamic_verify=True)
    backend = StubBackend(make_env(model), spec)
    assert backend.spec is spec
    assert (backend.spec.cache.page_size, backend.spec.causal, backend.spec.dynamic_verify) == (16, False, True)
    context = make_context(backend)
    assert isinstance(context, AttentionContext)
    assert isinstance(context.workspace, AttentionWorkspace)
    assert context.backend_identity is backend


@pytest.mark.parametrize("field,value", [("phase", AttentionPhase.PREFILL), ("family", AttentionFamily.MLA)])
def test_selection_rejects_mixed_specs(field, value):
    with pytest.raises(ValueError, match="share family, phase and cache layout"):
        selection.select_attention_backend(
            specs=(make_spec(), replace(make_spec(), **{field: value})),
            caps=model_stub().device_runtime.caps,
            execution=model_stub().attention_requirements,
        )


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("phase", list(AttentionPhase))
def test_gdn_output_contract(monkeypatch, replay, phase):
    from lightllm.common.attention.api import GDNInputs, GDNOutput
    from lightllm.common.attention.backends.gdn.replay.backend import TritonGDNReplayDecodeBackend
    from lightllm.common.attention.backends.gdn.triton.backend import TritonGDNDecodeBackend
    from unit_tests.common.attention.helpers import make_backend

    backend = make_backend(
        phase_class(TritonGDNReplayDecodeBackend if replay else TritonGDNDecodeBackend, phase), mtp_step=0
    )
    core, gate = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    monkeypatch.setattr(backend, "_split_qkvzba", lambda x: (x, gate, None, None))
    if phase is AttentionPhase.PREFILL:
        monkeypatch.setattr(backend, "_prefill" if replay else "_run_prefill", lambda *a: core)
    else:
        monkeypatch.setattr(backend, "_decode" if replay else "_run_decode", lambda *a: (core, gate))
    context = make_context(backend, make_batch(phase=phase))
    state = SimpleNamespace(finish_decode_token=lambda *a: None)
    layer = make_layer_attention(backend, context, SimpleNamespace(conv=None, ssm=None, replay_state=state))
    out = torch.empty_like(core)
    result = run_attention(layer, context, GDNInputs(core), out=out)
    assert isinstance(result, GDNOutput)
    assert result.hidden_states is out
    assert result.gate is gate
    torch.testing.assert_close(result.hidden_states, core)


def test_binding_resolves_context_on_each_forward(monkeypatch):
    from lightllm.common.attention.session import AttentionSession

    backend = _OutputBackend(make_env(), make_spec())
    first, second = make_context(backend), make_context(backend)
    layer = make_layer_attention(backend, first, params=None)
    session = AttentionSession()
    forwarded = []

    def forward(layer, context, inputs, *, out=None):
        forwarded.append(context)
        return inputs

    monkeypatch.setattr(backend, "forward", forward)
    result, out = torch.ones(2, 32), torch.empty(2, 32)
    for context in (first, second):
        session.contexts[0] = context
        assert layer.forward(session, result, out=out) is out
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

        def plan_host(self, context):
            pass

        def forward(self, layer, context, inputs, *, out=None):
            return inputs

    backend, session = Backend(), AttentionSession()
    batch = make_batch()
    capacity = ContextCapacity(phase=batch.phase, batch_size=2, token_num=2, max_kv_seq_len=16, max_q_seq_len=1)
    context = session.allocate(backend, capacity, capture=True, workspace=AttentionWorkspace("cpu"))
    session.plan(backend, context, batch)
    session.contexts[0] = context
    layer = make_layer_attention(StubBackend(make_env(), make_spec()), context, None)
    layer = replace(layer, backend=backend)
    out = torch.zeros(2, 4)
    assert layer.forward(session, torch.ones(2, 4), out=out) is out
    assert out.eq(1).all()
    with pytest.raises(ValueError, match="different attention backend"):
        replace(layer, backend=Backend()).forward(session, out)
    context.capacity = replace(context.capacity, phase=AttentionPhase.PREFILL)
    with pytest.raises(ValueError, match="different attention backend"):
        layer.forward(session, out)


def test_replay_copies_shared_batch_once(monkeypatch):
    from lightllm.common.attention.api import AttentionBatch
    from lightllm.common.attention.session import AttentionSession

    first, second = (_OutputBackend(make_env(), make_spec()) for _ in range(2))
    captured, source = AttentionSession(), AttentionSession()
    captured.groups = source.groups = ({AttentionPhase.DECODE: first}, {AttentionPhase.DECODE: second})
    captured.batch = make_batch()
    source.batch = make_batch((9, 7), req_to_token_indexs=captured.batch.req_to_token_indexs)
    for i, backend in enumerate((first, second)):
        captured.contexts[i] = make_context(backend, captured.batch, capture=True, max_length=16, prepare=True)
    copies = []
    original = AttentionBatch.input_copies

    def copy(batch, incoming):
        copies.append(batch)
        return original(batch, incoming)

    monkeypatch.setattr(AttentionBatch, "input_copies", copy)
    for target, incoming in captured.replay_input_copies(source):
        target.copy_(incoming)
    captured.plan_replay(source)
    assert len(copies) == 1
    for context in captured.contexts.values():
        assert context.batch.b_seq_len.tolist() == [9, 7]
        assert context.current_max_kv_seq_len == 9
        assert context.batch.max_kv_seq_len == 16


@pytest.mark.parametrize("failure", ["allocate", "plan", "forward"])
def test_session_releases_workspace_on_error(failure):
    from lightllm.common.attention.api import AttentionWorkspace, ContextCapacity
    from lightllm.common.attention.session import AttentionSession
    from lightllm.platform import DeviceRuntime

    events = []

    class Workspace(AttentionWorkspace):
        def begin(self):
            events.append("begin")

        def finish(self, dependencies=()):
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
    capacity = ContextCapacity(phase=batch.phase, batch_size=2, token_num=2, max_kv_seq_len=16, max_q_seq_len=1)
    with pytest.raises(RuntimeError, match=failure):
        with session.prepare(
            batch, ({batch.phase: backend},), capacity, workspace=lambda _: Workspace(runtime=DeviceRuntime("cpu"))
        ):
            raise RuntimeError("forward failed")
    assert events == ["begin", "finish"]


def test_model_capture_session_uses_graph_bound_and_one_batch():
    from lightllm.common.attention.api import AttentionWorkspace
    from lightllm.common.attention.session import AttentionSession
    from lightllm.common.basemodel.attention import ModelAttention
    from lightllm.platform import DeviceRuntime

    events = []

    class Workspace(AttentionWorkspace):
        def __init__(self, group_id):
            super().__init__(runtime=DeviceRuntime("cpu"))
            self.group_id = group_id

        def begin(self):
            events.append(("begin", self.group_id))

        def finish(self, dependencies=()):
            events.append(("finish", self.group_id))

    attention = ModelAttention.__new__(ModelAttention)
    attention.generation = 3
    attention.constraints = SimpleNamespace(max_seq_len=128)
    attention.groups = tuple({AttentionPhase.DECODE: StubBackend(make_env(), make_spec())} for _ in range(2))
    attention.workspaces = {(i, 0): Workspace(i) for i in range(2)}
    batch = make_batch((9, 7))
    state = SimpleNamespace(
        **vars(batch),
        is_prefill=False,
        input_ids=torch.zeros(batch.token_num, dtype=torch.int64),
        req_manager=SimpleNamespace(req_to_token_indexs=batch.req_to_token_indexs, HOLD_REQUEST_ID=batch.hold_req_id),
        att_state_manager=SimpleNamespace(storage_generation=3),
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

    assert events == [("begin", 0), ("begin", 1), ("finish", 1), ("finish", 0)]
