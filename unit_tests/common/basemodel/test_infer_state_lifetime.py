"""An InferStateInfo and its attention attention state are freed by reference counting, without cycle GC."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.infer_struct import InferStateInfo


class _AttState:
    def __init__(self, positions):
        self.positions = positions


class _Groups:
    """Builds one group's attention state from the step's tensors, as step builders do."""

    def build_attention_states(self, infer_state):
        return {"group": _AttState(infer_state.position_ids)}


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


def _create_state(is_prefill):
    owner = InferStateInfo()
    owner.is_prefill = is_prefill
    owner.position_ids = torch.arange(4)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "lightllm.common.basemodel.infer_struct.build_attention_states",
            lambda model, state: _Groups().build_attention_states(state),
        )
        owner.init_att_state(SimpleNamespace(attention_step_groups={}))
    return owner


@pytest.mark.parametrize("is_prefill", [False, True])
def test_att_state_do_not_keep_their_infer_state_alive(no_cycle_gc, is_prefill):
    owner = _create_state(is_prefill)
    att_state = owner.att_states["group"]
    owner_ref = weakref.ref(owner)
    tensor_ref = weakref.ref(owner.position_ids)

    # An external reference to the attention state must not retain their InferStateInfo.
    del owner
    assert owner_ref() is None
    assert tensor_ref() is not None
    att_state_ref = weakref.ref(att_state)
    del att_state
    assert att_state_ref() is None
    assert tensor_ref() is None


def test_infer_state_without_attention_builds_no_att_state():
    owner = InferStateInfo()
    owner.is_prefill = False
    owner.init_att_state(SimpleNamespace(attention_step_groups=None))
    assert owner.att_states == {}


def test_decode_graph_replay_preserves_capture_buffers_and_owner_lifetime(no_cycle_gc):
    captured = _create_state(is_prefill=False)
    captured.input_ids = torch.zeros(4, dtype=torch.int64)
    captured_ref = weakref.ref(captured)
    capture_positions_ptr = captured.position_ids.data_ptr()
    replay_calls = []

    graph = CudaGraph.__new__(CudaGraph)
    output = object()
    graph.graph = {4: (SimpleNamespace(replay=lambda: replay_calls.append(True)), captured, output)}
    del captured
    assert captured_ref() is not None

    runtime = _create_state(is_prefill=False)
    runtime.input_ids = torch.ones(4, dtype=torch.int64)
    runtime.position_ids.add_(7)
    runtime_ref = weakref.ref(runtime)
    assert graph._replay(runtime) is output
    assert replay_calls == [True]
    # Replay copies the step's tensors into the captured state's, which its attention state read.
    assert captured_ref().position_ids.data_ptr() == capture_positions_ptr
    assert captured_ref().att_states["group"].positions is captured_ref().position_ids
    torch.testing.assert_close(captured_ref().position_ids, runtime.position_ids)

    del runtime
    assert runtime_ref() is None
    graph.graph.clear()
    assert captured_ref() is None
