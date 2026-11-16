from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.basemodel as basemodel_module
from lightllm.common.basemodel.attention import AttBackendSlot, BaseAttBackend, LINEAR_SLOT, MAIN_SLOT
from lightllm.common.basemodel.attention.base_att import BaseDecodeAttState, BasePrefillAttState
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.infer_struct import InferStateInfo


class _State:
    def __init__(self, backend, infer_state):
        self.backend = backend
        self.init_calls = 0
        self.copied_from = []

    def init_state(self):
        self.init_calls += 1

    def check_no_infer_state(self):
        pass

    def copy_for_decode_cuda_graph(self, new_state):
        self.copied_from.append(new_state)


class _Backend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state):
        return _State(self, infer_state)

    def create_att_decode_state(self, infer_state):
        return _State(self, infer_state)


class _OtherBackend(_Backend):
    pass


def test_backends_are_not_shared_between_constructions():
    model = object()

    assert _Backend(model=model) is not _Backend(model=model)


def test_slot_shares_one_instance_when_both_phases_use_the_same_class():
    model = object()

    slot = AttBackendSlot.build(model, _Backend, _Backend)

    assert slot.prefill is slot.decode
    assert slot.prefill.model is model
    assert slot.backends() == (slot.prefill,)


def test_slot_creates_separate_instances_for_different_classes():
    slot = AttBackendSlot.build(object(), _Backend, _OtherBackend)

    assert type(slot.prefill) is _Backend and type(slot.decode) is _OtherBackend
    assert slot.backends() == (slot.prefill, slot.decode)


def test_slot_phase_without_backend_is_none():
    slot = AttBackendSlot.build(object(), prefill_class=None, decode_class=_Backend)

    assert slot.prefill is None
    assert slot.backends() == (slot.decode,)
    assert AttBackendSlot().backends() == ()


def _model_with_slots(slots):
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.is_mtp_draft_model = False
    model._init_att_backends = lambda: slots
    model.init_att_backends()
    return model


def test_model_exposes_main_slot_backends():
    main = AttBackendSlot.build(object(), _Backend, _OtherBackend)
    model = _model_with_slots({MAIN_SLOT: main, LINEAR_SLOT: AttBackendSlot.build(object(), _Backend, _Backend)})

    assert model.prefill_att_backend is main.prefill
    assert model.decode_att_backend is main.decode
    assert list(model.att_backends) == [MAIN_SLOT, LINEAR_SLOT]


def test_init_att_backends_rebuilds_instances():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.is_mtp_draft_model = False
    model._init_att_backends = lambda: {MAIN_SLOT: AttBackendSlot.build(model, _Backend, _Backend)}

    model.init_att_backends()
    first = model.prefill_att_backend
    model.init_att_backends()

    assert model.prefill_att_backend is not first


def test_default_model_resolves_the_dense_family_for_the_main_slot(monkeypatch):
    requested = []

    def fake_resolve(family, phase, **kwargs):
        requested.append((family, phase, kwargs))
        return _Backend

    monkeypatch.setattr(basemodel_module, "resolve_att_backend_class", fake_resolve)
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.is_mtp_draft_model = False
    model.att_state_manager = SimpleNamespace(get_full_attention_states=lambda role: [object()])

    slots = model._init_att_backends()

    assert requested == [("dense", "prefill", {}), ("dense", "decode", {})]
    assert list(slots) == [MAIN_SLOT]
    assert slots[MAIN_SLOT].prefill is slots[MAIN_SLOT].decode


def test_model_without_full_attention_states_has_no_main_slot(monkeypatch):
    monkeypatch.setattr(
        basemodel_module,
        "resolve_att_backend_class",
        lambda *args, **kwargs: pytest.fail("no backend should be selected without full-attention states"),
    )
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.is_mtp_draft_model = False
    model.att_state_manager = SimpleNamespace(get_full_attention_states=lambda role: [])

    model.init_att_backends()

    assert model.att_backends == {}
    assert model.prefill_att_backend is None and model.decode_att_backend is None


@pytest.mark.parametrize("is_prefill", [True, False])
def test_infer_state_only_gets_states_for_its_phase(is_prefill):
    infer_state = InferStateInfo()
    infer_state.is_prefill = is_prefill
    slots = {
        MAIN_SLOT: AttBackendSlot.build(object(), prefill_class=None, decode_class=_Backend),
        "full": AttBackendSlot.build(object(), _Backend, _Backend),
    }

    # Same loop as TpPartBaseModel._create_inferstate, which needs a full model to run.
    TpPartBaseModel._create_att_states(SimpleNamespace(att_backends=slots), infer_state)

    if is_prefill:
        # The main slot has no prefill backend, so it gets no prefill state.
        assert list(infer_state.prefill_att_states) == ["full"]
        assert infer_state.decode_att_states == {}
        assert infer_state.prefill_att_state is None
        assert infer_state.prefill_att_states["full"].backend is slots["full"].prefill
    else:
        assert list(infer_state.decode_att_states) == [MAIN_SLOT, "full"]
        assert infer_state.prefill_att_states == {}
        assert infer_state.decode_att_state is infer_state.decode_att_states[MAIN_SLOT]
    infer_state.init_att_state()
    states = infer_state.prefill_att_states if is_prefill else infer_state.decode_att_states
    assert [state.init_calls for state in states.values()] == [1] * len(states)


def test_main_state_attributes_are_views_of_the_main_slot():
    infer_state = InferStateInfo()
    assert infer_state.prefill_att_state is None and infer_state.decode_att_state is None

    prefill_state, decode_state = object(), object()
    infer_state.prefill_att_state = prefill_state
    infer_state.decode_att_state = decode_state

    assert infer_state.prefill_att_states == {MAIN_SLOT: prefill_state}
    assert infer_state.decode_att_states == {MAIN_SLOT: decode_state}

    infer_state.prefill_att_state = None
    assert infer_state.prefill_att_states == {}


def test_cuda_graph_copy_refreshes_every_decode_slot_from_the_matching_slot():
    graph_state, new_state = InferStateInfo(), InferStateInfo()
    for infer_state in (graph_state, new_state):
        infer_state.decode_att_states = {MAIN_SLOT: _State(None, infer_state), LINEAR_SLOT: _State(None, infer_state)}
    graph_state.b_seq_len = torch.tensor([1, 2])
    new_state.b_seq_len = torch.tensor([3, 4])

    graph_state.copy_for_cuda_graph(new_state)

    assert graph_state.b_seq_len.tolist() == [3, 4]
    for slot_name in (MAIN_SLOT, LINEAR_SLOT):
        assert graph_state.decode_att_states[slot_name].copied_from == [new_state.decode_att_states[slot_name]]
