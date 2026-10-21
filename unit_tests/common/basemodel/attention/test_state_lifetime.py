import gc
import weakref

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import BaseDecodeAttState, BasePrefillAttState
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.infer_struct import InferStateInfo


class _PrefillState(BasePrefillAttState):
    def init_state(self):
        self.positions = self.infer_state.position_ids

    def prefill_att(self, q, k, v, **kwargs):
        return q


class _DecodeState(BaseDecodeAttState):
    def init_state(self):
        self.positions = self.infer_state.position_ids

    def decode_att(self, q, k, v, **kwargs):
        return q


@pytest.fixture
def no_cycle_gc():
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()


def _create_state(is_prefill, secondary=False):
    owner = InferStateInfo()
    owner.is_prefill = is_prefill
    owner.position_ids = torch.arange(4)
    state_cls = _PrefillState if is_prefill else _DecodeState
    slot = "prefill_att_state" if is_prefill else "decode_att_state"
    setattr(owner, slot, state_cls(_infer_state_weak_ref=weakref.ref(owner)))
    if secondary:
        setattr(owner, slot + "1", state_cls(_infer_state_weak_ref=weakref.ref(owner)))
    owner.init_att_state()
    return owner


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("secondary", [False, True])
def test_forward_states_release_without_cycle_gc(no_cycle_gc, is_prefill, secondary):
    owner = _create_state(is_prefill, secondary)
    states = [state for state in vars(owner).values() if isinstance(state, (BasePrefillAttState, BaseDecodeAttState))]
    owner_ref = weakref.ref(owner)
    tensor_ref = weakref.ref(owner.position_ids)
    state_refs = [weakref.ref(state) for state in states]

    # Even an external reference to attention metadata must not retain its owner.
    del owner
    assert owner_ref() is None
    with pytest.raises(ReferenceError):
        states[0].infer_state
    assert tensor_ref() is not None
    del states
    assert all(ref() is None for ref in state_refs)
    assert tensor_ref() is None


@pytest.mark.parametrize("state_cls", [_PrefillState, _DecodeState])
def test_attention_getter_uses_a_weak_reference_and_is_read_only(state_cls):
    state = state_cls()
    assert state.infer_state is None

    owner = InferStateInfo()
    state = state_cls(_infer_state_weak_ref=weakref.ref(owner))
    assert isinstance(state._infer_state_weak_ref, weakref.ReferenceType)
    assert state.infer_state is owner
    state.check_no_infer_state()
    with pytest.raises(AttributeError):
        state.infer_state = owner


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("slot_suffix", ["", "1"])
def test_check_rejects_infer_state_under_any_attribute_name(is_prefill, slot_suffix):
    owner = _create_state(is_prefill, secondary=True)
    slot = ("prefill_att_state" if is_prefill else "decode_att_state") + slot_suffix
    attention = getattr(owner, slot)

    class _OtherInferState(InferStateInfo):
        pass

    # Check the type, rather than just the identity of the expected owner.
    attention.saved_context = _OtherInferState()
    with pytest.raises(RuntimeError, match="saved_context"):
        owner.check_att_states()


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("slot_suffix", ["", "1"])
def test_init_checks_for_accidental_back_reference(is_prefill, slot_suffix):
    owner = _create_state(is_prefill, secondary=True)
    slot = ("prefill_att_state" if is_prefill else "decode_att_state") + slot_suffix
    base = _PrefillState if is_prefill else _DecodeState

    class _RetainingState(base):
        def init_state(self):
            self.saved_context = self.infer_state

    setattr(owner, slot, _RetainingState(_infer_state_weak_ref=weakref.ref(owner)))
    try:
        with pytest.raises(RuntimeError, match="saved_context"):
            owner.init_att_state()
    finally:
        del getattr(owner, slot).saved_context


def test_decode_graph_replay_preserves_capture_buffers_and_owner_lifetime(no_cycle_gc):
    captured = _create_state(is_prefill=False, secondary=True)
    captured.input_ids = torch.zeros(4, dtype=torch.int64)
    captured_ref = weakref.ref(captured)
    capture_positions_ptr = captured.position_ids.data_ptr()
    replay_calls = []

    class _Graph:
        def replay(self):
            replay_calls.append(True)

    graph = CudaGraph.__new__(CudaGraph)
    output = object()
    graph.graph = {4: (_Graph(), captured, output)}
    del captured
    assert captured_ref() is not None

    runtime = _create_state(is_prefill=False, secondary=True)
    runtime.input_ids = torch.ones(4, dtype=torch.int64)
    runtime.position_ids.add_(7)
    runtime_ref = weakref.ref(runtime)
    assert graph._replay(runtime) is output
    assert replay_calls == [True]
    assert captured_ref().position_ids.data_ptr() == capture_positions_ptr
    assert captured_ref().decode_att_state.infer_state is captured_ref()
    assert captured_ref().decode_att_state1.infer_state is captured_ref()
    torch.testing.assert_close(captured_ref().position_ids, runtime.position_ids)

    del runtime
    assert runtime_ref() is None
    graph.graph.clear()
    assert captured_ref() is None
