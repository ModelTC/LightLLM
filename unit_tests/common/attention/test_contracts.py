from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import selection
from lightllm.common.attention.api import AttentionFamily, AttentionPhase, UnsupportedAttention
from lightllm.common.attention.backends.base import BaseAttentionBackend
from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, make_spec, model_stub


def _select(*, model, specs=None, **kwargs):
    return selection.select_attention_backend(
        specs=specs or (make_spec(kwargs["family"], kwargs["phase"]),),
        caps=model.device_runtime.caps,
        execution=model.attention_execution,
        encoding=model.args.llm_kv_type,
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
    backend = BaseAttentionBackend(model_stub())
    context = make_context(backend, capture=capture, max_length=16)
    backend.prepare(context, make_batch(req_to_token_indexs=context.batch.req_to_token_indexs))
    with pytest.raises(ValueError, match="capacity"):
        backend.prepare(context, make_batch((17, 2)))
    with pytest.raises(ValueError, match="different attention backend"):
        BaseAttentionBackend(model_stub()).prepare(context, make_batch())
    with pytest.raises(ValueError, match="different attention backend"):
        BaseAttentionBackend(model_stub()).update(context)
    attention = make_layer_attention(BaseAttentionBackend(model_stub()), context, params=None)
    with pytest.raises(ValueError, match="different attention backend"):
        attention.forward(context, inputs=None)


def test_layer_attention_rejects_wrong_phase():
    backend = BaseAttentionBackend(model_stub())
    context = make_context(backend, prepare=True)
    attention = make_layer_attention(backend, context, params=None)
    attention = replace(attention, spec=replace(attention.spec, phase=AttentionPhase.PREFILL))
    with pytest.raises(ValueError, match="phases differ"):
        attention.forward(context, inputs=None)


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

    monkeypatch.setattr("lightllm.common.attention.selection.validate", lambda *args, **kwargs: True)
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
    monkeypatch.setattr(selection, "validate", lambda *args, **kwargs: pytest.fail("probed rejected inputs"))
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
    candidates = selection.data_type_to_backend[format]
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
    backend = _OutputBackend(model_stub())
    context = make_context(backend, prepare=True)
    layer = make_layer_attention(backend, context, params=None)
    out = torch.empty(2, 8 if strided else 4, 8)[:, :4]
    gate = torch.ones(2)
    output, actual_gate = layer.forward(context, (torch.ones(2, 32), gate), out=out)
    assert output is out and actual_gate is gate
    assert torch.all(out == 1)
    valid_heads = out[:, :2]
    output, actual_gate = layer.forward(context, (valid_heads, gate), out=out)
    assert output is valid_heads and actual_gate is gate


@pytest.mark.parametrize("invalid", ["dtype", "device", "size", "view_bounds"])
def test_output_rejects_incompatible_buffer(invalid):
    backend = _OutputBackend(model_stub())
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
        layer.forward(context, result, out=out)


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
    backend = BaseAttentionBackend(model_stub())
    context = make_context(backend, borrowed, prepare=True)
    state.attention_contexts = {0: context}
    layer = make_layer_attention(backend, context, params=None)
    assert layer.context(state) is context
    with pytest.raises(ValueError, match="binding"):
        replace(layer, backend=BaseAttentionBackend(model_stub())).context(state)
    with pytest.raises(AttributeError):
        context.b_requ_idx = borrowed.b_req_idx
