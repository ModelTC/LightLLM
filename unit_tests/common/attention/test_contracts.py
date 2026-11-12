from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import selection
from lightllm.common.attention.api import AttentionFamily, AttentionPhase, UnsupportedAttention
from lightllm.common.attention.backends.base import BaseAttentionBackend
from unit_tests.common.attention.helpers import (
    make_backend,
    make_batch,
    make_layer_attention,
    make_plan,
    make_spec,
    model_stub,
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
def test_plan_capacity_and_backend_identity(capture):
    backend = make_backend()
    plan = make_plan(backend, capture=capture, max_length=16)
    backend.prepare(plan, make_batch(req_to_token_indexs=plan.batch.req_to_token_indexs))
    with pytest.raises(ValueError, match="capacity"):
        backend.prepare(plan, make_batch((17, 2)))
    with pytest.raises(ValueError, match="different attention backend"):
        make_backend().prepare(plan, make_batch())
    with pytest.raises(ValueError, match="different attention backend"):
        make_backend().update(plan)
    attention = make_layer_attention(make_backend(), plan, params=None)
    with pytest.raises(ValueError, match="different attention backend"):
        attention.forward(plan, inputs=None)


def test_layer_attention_rejects_wrong_phase():
    backend = make_backend()
    plan = make_plan(backend, prepare=True)
    attention = make_layer_attention(backend, plan, params=None)
    attention = replace(attention, spec=replace(attention.spec, phase=AttentionPhase.PREFILL))
    with pytest.raises(ValueError, match="phases differ"):
        attention.forward(plan, inputs=None)


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_selection_never_falls_back_to_a_rejected_backend(monkeypatch, requested):
    spec = make_spec(cache=replace(make_spec().cache, format="fp8kv_sph"))
    monkeypatch.setattr("lightllm.common.basemodel.attention.describe_attention_layers", lambda *args: (spec,))
    monkeypatch.setattr(
        BaseAttentionBackend,
        "check_environment",
        classmethod(lambda cls, caps: None),
    )
    with pytest.raises(selection.BackendSelectionError, match="encoding fp8kv_sph"):
        selection.select_attention_backend(
            family=spec.family,
            phase=spec.phase,
            model=model_stub(),
            requested=requested,
            candidates={"None": {"triton": BaseAttentionBackend}},
            priority=["triton"],
        )


@pytest.mark.parametrize("requested", ["auto", "triton"])
def test_environment_checks_apply_to_explicit_and_auto(monkeypatch, requested):
    monkeypatch.setattr("lightllm.common.basemodel.attention.describe_attention_layers", lambda *args: (make_spec(),))

    def reject(cls, caps):
        raise UnsupportedAttention("missing extension")

    monkeypatch.setattr(BaseAttentionBackend, "check_environment", classmethod(reject))
    with pytest.raises(selection.BackendSelectionError, match="missing extension"):
        selection.select_attention_backend(
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

    monkeypatch.setattr("lightllm.common.basemodel.attention.describe_attention_layers", lambda *args: (make_spec(),))
    monkeypatch.setattr("lightllm.common.attention.selection.validate", lambda *args, **kwargs: True)
    return {"None": {"broken": BrokenBackend, "working": WorkingBackend}}


@pytest.mark.parametrize("error_type", [UnsupportedAttention, ValueError, RuntimeError])
def test_backend_check_exception_allows_auto_fallback(monkeypatch, error_type):
    candidates = _backend_check_exception_candidates(monkeypatch, error_type)
    selected = selection.select_attention_backend(
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
        selection.select_attention_backend(
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
        selection.select_attention_backend(
            family=spec.family,
            phase=spec.phase,
            model=model_stub(),
            requested=requested,
            specs=(spec,),
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

    original = make_backend(FlashInferAttBackend)
    small = original.for_spec(make_spec(num_q_heads=8, head_size=64))
    large = original.for_spec(make_spec(num_q_heads=16, head_size=128))
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
        return lambda model: SimpleNamespace(for_spec=lambda spec: SimpleNamespace(spec=spec))

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
