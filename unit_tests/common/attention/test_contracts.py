from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import registry, selection
from lightllm.common.attention.api import AttentionFamily, AttentionPhase, UnsupportedAttention
from lightllm.common.attention.backends.base import BaseAttentionBackend
from lightllm.utils.backend_validator import ProbeResult, ProbeScope
from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, make_spec, model_stub


def _select(*, model, specs=None, **kwargs):
    family, phase = kwargs.pop("family"), kwargs.pop("phase")
    return selection.select_attention_backend(
        specs=specs or (make_spec(family, phase),),
        caps=model.device_runtime.caps,
        execution=model.attention_requirements,
        ssm_state_mode=model.args.ssm_state_mode,
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
    backend = BaseAttentionBackend(model_stub(), make_spec())
    context = make_context(backend, capture=capture, max_length=16)
    backend.prepare(context, make_batch(req_to_token_indexs=context.batch.req_to_token_indexs))
    with pytest.raises(ValueError, match="capacity"):
        backend.prepare(context, make_batch((17, 2)))
    with pytest.raises(ValueError, match="different attention backend"):
        BaseAttentionBackend(model_stub(), make_spec()).prepare(context, make_batch())
    with pytest.raises(ValueError, match="different attention backend"):
        BaseAttentionBackend(model_stub(), make_spec()).update(context)
    attention = make_layer_attention(BaseAttentionBackend(model_stub(), make_spec()), context, params=None)
    with pytest.raises(ValueError, match="different attention backend"):
        attention.backend.forward(attention, context, inputs=None)


def test_layer_attention_rejects_wrong_phase():
    backend = BaseAttentionBackend(model_stub(), make_spec())
    context = make_context(backend, prepare=True)
    attention = make_layer_attention(backend, context, params=None)
    attention = replace(attention, spec=replace(attention.spec, phase=AttentionPhase.PREFILL))
    with pytest.raises(ValueError, match="phases differ"):
        attention.backend.forward(attention, context, inputs=None)


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_selection_never_falls_back_to_a_rejected_backend(monkeypatch, requested):
    spec = make_spec(cache=replace(make_spec().cache, format="fp8kv_sph"))
    monkeypatch.setattr(
        BaseAttentionBackend,
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
            candidates={"None": {"triton": BaseAttentionBackend}},
            priority=["triton"],
        )


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_environment_checks_apply_to_explicit_and_auto(monkeypatch, requested):
    def reject(cls, caps):
        raise UnsupportedAttention("missing extension")

    monkeypatch.setattr(BaseAttentionBackend, "check_environment", classmethod(reject))
    with pytest.raises(selection.BackendSelectionError, match="missing extension"):
        _select(
            family=AttentionFamily.DENSE,
            phase=AttentionPhase.DECODE,
            model=model_stub(),
            requested=requested,
            candidates={"None": {"triton": BaseAttentionBackend}},
            priority=["triton"],
        )


def _backend_check_exception_candidates(monkeypatch, error_type=RuntimeError):
    class BrokenBackend(BaseAttentionBackend):
        provider = "broken"

        @classmethod
        def check_environment(cls, caps):
            raise error_type("provider ABI mismatch")

    class WorkingBackend(BaseAttentionBackend):
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
    monkeypatch.setattr(BaseAttentionBackend, "check_environment", classmethod(lambda cls, caps: None))
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
            candidates={"None": {"triton": BaseAttentionBackend}},
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
    candidates = registry.cache_format_to_backend[format]
    assert provider not in candidates


def test_layer_shape_configures_group_planner():
    from lightllm.common.attention.backends.dense.flashinfer.backend import FlashInferAttBackend

    model = model_stub(page_size=16)
    model.config = {"num_attention_heads": 16, "num_key_value_heads": 4, "hidden_size": 2048}
    model.max_seq_length = 128
    model.data_type = torch.bfloat16
    small = FlashInferAttBackend.create(model, make_spec(num_q_heads=8, head_size=64))
    large = FlashInferAttBackend.create(model, make_spec(num_q_heads=16, head_size=128))
    assert (small.tp_q_head_num, small.head_dim) == (8, 64)
    assert (large.tp_q_head_num, large.head_dim) == (16, 128)


def test_groups_follow_each_layers_family_and_shape(monkeypatch):
    from lightllm.common.basemodel import attention

    specs = [
        make_spec(num_q_heads=8, head_size=64),
        make_spec(num_q_heads=8, head_size=64),
        make_spec(num_q_heads=16, head_size=128),
        make_spec(AttentionFamily.GDN),
    ]
    calls = []

    def select(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(create=lambda model, spec: SimpleNamespace())

    monkeypatch.setattr(selection, "select_attention_backend", select)
    monkeypatch.setattr(
        attention, "describe_attention_layer", lambda model, layer, family, phase: replace(layer.spec, phase=phase)
    )
    model = model_stub()
    model.attention_backend_candidates = lambda family, phase: (None, None)
    model.layers_infer = [SimpleNamespace(spec=spec, get_attention_family=lambda s=spec: s.family) for spec in specs]
    attention.initialize_attention_groups(model)
    assert model._attention_layer_groups == (0, 0, 1, 2)
    assert len(calls) == 6
    assert [group.prefill_spec.head_size for group in model.attention_groups[:2]] == [64, 128]
    assert model.attention_groups[-1].decode_spec.family is AttentionFamily.GDN


@pytest.mark.parametrize("invalid", ["input_ids", "group", "role"])
def test_runner_graph_copy_rejects_before_mutating_batch(invalid):
    from lightllm.common.basemodel.infer_struct import InferStateInfo

    captured, incoming = InferStateInfo(), InferStateInfo()
    captured.att_state_role = incoming.att_state_role = "main"
    group = object()
    captured.attention_groups = incoming.attention_groups = (group,)
    captured.attention_batch = make_batch()
    incoming.attention_batch = make_batch((9, 7), req_to_token_indexs=captured.attention_batch.req_to_token_indexs)
    captured.input_ids = torch.zeros(2, dtype=torch.long)
    incoming.input_ids = torch.ones(3 if invalid == "input_ids" else 2, dtype=torch.long)
    if invalid == "group":
        incoming.attention_groups = (object(),)
    if invalid == "role":
        incoming.att_state_role = "draft"
    before = captured.attention_batch.b_seq_len.clone()
    with pytest.raises(ValueError):
        captured.copy_for_cuda_graph(incoming)
    torch.testing.assert_close(captured.attention_batch.b_seq_len, before)
    assert captured.input_ids.tolist() == [0, 0]


def test_replay_flashqla_rejects_decode_and_fp32():
    from lightllm.common.attention.backends.gdn.replay.backend import ReplayGDNFlashQlaAttBackend

    spec = make_spec(family=AttentionFamily.GDN)
    with pytest.raises(UnsupportedAttention, match="decode"):
        ReplayGDNFlashQlaAttBackend.check_support(spec)
    with pytest.raises(UnsupportedAttention, match="dtype"):
        ReplayGDNFlashQlaAttBackend.check_support(replace(spec, phase=AttentionPhase.PREFILL, dtype=torch.float32))


class _OutputBackend(BaseAttentionBackend):
    def _forward_decode(self, layer, context, inputs, *, out=None):
        return inputs


@pytest.mark.parametrize("strided", [False, True])
def test_output_copy_and_padded_view_preserve_auxiliary_result(strided):
    backend = _OutputBackend(model_stub(), make_spec())
    context = make_context(backend, prepare=True)
    layer = make_layer_attention(backend, context, params=None)
    out = torch.empty(2, 8 if strided else 4, 8)[:, :4]
    gate = torch.ones(2)
    output, actual_gate = layer.backend.forward(layer, context, (torch.ones(2, 32), gate), out=out)
    assert output is out and actual_gate is gate
    assert torch.all(out == 1)
    valid_heads = out[:, :2]
    output, actual_gate = layer.backend.forward(layer, context, (valid_heads, gate), out=out)
    assert output is valid_heads and actual_gate is gate


@pytest.mark.parametrize("invalid", ["dtype", "device", "size", "view_bounds"])
def test_output_rejects_incompatible_buffer(invalid):
    backend = _OutputBackend(model_stub(), make_spec())
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
        layer.backend.forward(layer, context, result, out=out)


def test_batch_builder_borrows_runtime_inputs_and_binding_checks_owner():
    from lightllm.common.basemodel.attention_batch import AttentionBatchBuilder
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
    borrowed = AttentionBatchBuilder.from_infer_state(state)
    assert borrowed.b_req_idx is state.b_req_idx
    assert borrowed.b1_cu_q_seq_len is state.b1_cu_q_seq_len
    assert borrowed.req_to_token_indexs is state.req_manager.req_to_token_indexs
    assert borrowed.b_image_token_end is state.b_image_token_end
    backend = _OutputBackend(model_stub(), make_spec())
    context = make_context(backend, borrowed, prepare=True)
    state.attention_contexts = {0: context}
    layer = make_layer_attention(backend, context, params=None)
    assert layer.context(state) is context
    output = torch.ones(2, 32)
    assert layer.forward(state, output) is output
    with pytest.raises(ValueError, match="binding"):
        replace(layer, spec=replace(layer.spec, phase=AttentionPhase.PREFILL)).forward(state, output)
    with pytest.raises(ValueError, match="binding"):
        replace(layer, backend=_OutputBackend(model_stub(), make_spec())).forward(state, output)
    with pytest.raises(ValueError, match="binding"):
        replace(layer, backend=BaseAttentionBackend(model_stub(), make_spec())).context(state)
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
    result = validator._probe_in_process("selected", device="cpu", spec=replace(make_spec(), use_mtp=True))
    assert not result.ok
    assert result.scope is validator.ProbeScope.PROVIDER_SMOKE
    assert result.reason == "provider failed"


def test_replay_flashqla_uses_inherited_initialization(monkeypatch):
    import sys
    from types import ModuleType

    from lightllm.common.attention.backends.gdn.replay.backend import (
        ReplayGDNFlashQlaAttBackend,
        ReplayGDNTritonAttBackend,
    )

    calls = []
    original = ReplayGDNTritonAttBackend.__init__

    def initialize(self, model, spec):
        calls.append("parent")
        original(self, model, spec)

    monkeypatch.setattr(ReplayGDNTritonAttBackend, "__init__", initialize)
    monkeypatch.setattr(ReplayGDNTritonAttBackend, "_init_linear_layer_metadata", lambda *a: calls.append("metadata"))
    module = ModuleType("flash_qla")
    module.chunk_gated_delta_rule = object()
    monkeypatch.setitem(sys.modules, "flash_qla", module)
    model = model_stub()
    backend = ReplayGDNFlashQlaAttBackend(model, make_spec(AttentionFamily.GDN))
    assert calls == ["parent", "metadata"]
    assert backend.runtime is model.device_runtime
    assert backend.prefill_kernel is module.chunk_gated_delta_rule


def test_spec_configures_backend_before_planning():
    from lightllm.common.attention.api import AttentionExecutionContext, AttentionWorkspace

    model = model_stub(page_size=1, mtp_dynamic_verify=False)
    spec = replace(make_spec(), cache=replace(make_spec().cache, page_size=16), causal=False, dynamic_verify=True)
    backend = BaseAttentionBackend.create(model, spec)
    assert backend.spec is spec
    assert (backend.page_size, backend.causal, backend.dynamic_verify) == (16, False, True)
    context = make_context(backend)
    assert isinstance(context, AttentionExecutionContext)
    assert isinstance(context.workspace, AttentionWorkspace)
    assert context.backend_identity is backend


@pytest.mark.parametrize("field,value", [("phase", AttentionPhase.PREFILL), ("family", AttentionFamily.MLA)])
def test_selection_rejects_mixed_specs(field, value):
    with pytest.raises(ValueError, match="share family, phase and cache format"):
        selection.select_attention_backend(
            specs=(make_spec(), replace(make_spec(), **{field: value})),
            caps=model_stub().device_runtime.caps,
            execution=model_stub().attention_requirements,
        )


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("phase", list(AttentionPhase))
def test_gdn_output_contract(monkeypatch, replay, phase):
    from lightllm.common.attention.api import GDNInputs, GDNOutput
    from lightllm.common.attention.backends.gdn.replay.backend import ReplayGDNTritonAttBackend
    from lightllm.common.attention.backends.gdn.triton.backend import TritonGDNBackend
    from unit_tests.common.attention.helpers import make_backend

    backend = make_backend(ReplayGDNTritonAttBackend if replay else TritonGDNBackend, mtp_step=0)
    core, gate = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    monkeypatch.setattr(backend, "_split_qkvzba", lambda x: (x, gate, None, None))
    monkeypatch.setattr(backend, "_prefill" if replay else "_run_prefill", lambda *a: core)
    monkeypatch.setattr(backend, "_decode" if replay else "_run_decode", lambda *a: (core, gate))
    context = make_context(backend, make_batch(phase=phase))
    state = SimpleNamespace(finish_decode_token=lambda *a: None)
    layer = make_layer_attention(backend, context, SimpleNamespace(conv=None, ssm=None, replay_state=state))
    out = torch.empty_like(core)
    result = layer.backend.forward(layer, context, GDNInputs(core), out=out)
    assert isinstance(result, GDNOutput)
    assert result.hidden_states is out
    assert result.gate is gate
    torch.testing.assert_close(result.hidden_states, core)


def test_binding_resolves_context_on_each_forward(monkeypatch):
    backend = _OutputBackend(model_stub(), make_spec())
    first, second = make_context(backend), make_context(backend)
    binding = make_layer_attention(backend, first, params=None)
    contexts = {binding.group_id: first}
    seen = []

    def resolve(layer):
        assert layer is binding
        seen.append(contexts[layer.group_id])
        return contexts[layer.group_id]

    forwarded = []

    def forward(layer, context, inputs, *, out=None):
        assert layer is binding
        forwarded.append(context)
        return inputs

    monkeypatch.setattr(backend, "_forward_decode", forward)
    execution = SimpleNamespace(attention_context=resolve)
    result = torch.ones(2, 32)
    out = torch.empty_like(result)
    assert binding.forward(execution, result, out=out) is out
    contexts[binding.group_id] = second
    assert binding.forward(execution, result, out=out) is out
    assert seen == [first, second]
    assert all(actual is expected for actual, expected in zip(forwarded, (first, second)))
    torch.testing.assert_close(out, result)
