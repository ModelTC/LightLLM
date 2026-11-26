"""An InferStateInfo and its attention attention state are freed by reference counting, without cycle GC."""
from typing import Any
from unit_tests.common.attention.backends.step_fixture import create_att_state

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.attention.attention_backend_manager import AttentionBackendManager
from lightllm.common.attention.base_backend import AttRuntimeConfig, DecodeAttBackend, PrefillAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.attention.backends.gqa import GQAAttConfig
from lightllm.common.attention.backends.gqa.triton import TritonGQAPrefillBackend
from lightllm.common.attention.backends.mla import MLAPrefillAttConfig
from lightllm.common.attention.backends.mla.triton import TritonMLAPrefillBackend
from lightllm.common.attention.backends.sparse_mla import SparseMLAAttConfig
from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLAPrefillBackend


def test_sparse_scheduler_is_released_with_its_step(no_cycle_gc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from lightllm.common.basemodel.infer_struct import GraphMode
    from lightllm.common.attention.backends.sparse_mla.flashmla import FlashMLASparseDecodeBackend

    class Scheduler:
        pass

    monkeypatch.setitem(sys.modules, "flash_mla", SimpleNamespace(get_mla_metadata=lambda: (Scheduler(), None)))
    backend = FlashMLASparseDecodeBackend(
        att_config=SparseMLAAttConfig(
            num_heads=64,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=0.1,
            dtype=torch.bfloat16,
            kv_cache_format="fp8_ds_mla",
        )
    )
    owner = InferStateInfo()
    owner.graph_mode = GraphMode.CAPTURE
    state = create_att_state(backend=backend, infer_state=owner)
    scheduler = weakref.ref(state.sched_meta)
    del state, owner
    assert scheduler() is None


class _AttState:
    def __init__(self, positions):
        self.positions = positions
        self.derived = positions * 2


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


def _runtime(*backends: Any) -> AttentionBackendManager:
    runtime = AttentionBackendManager()
    runtime.register_backends(role="main", backends=backends)
    return runtime


def _create_state(is_prefill: bool, backend: PrefillAttBackend | DecodeAttBackend | None = None) -> InferStateInfo:
    owner = InferStateInfo()
    owner.is_prefill = is_prefill
    owner.position_ids = torch.arange(4)
    owner.b_req_idx = owner.position_ids
    owner.req_manager = SimpleNamespace(req_to_token_indexs=None)

    class Backend(PrefillAttBackend if is_prefill else DecodeAttBackend):
        @staticmethod
        def get_name():
            return "test"

        def create_att_state(self, infer_state: Any):
            return _AttState(positions=infer_state.b_req_idx)

        def prefill(self, *args: Any, **kwargs: Any):
            raise NotImplementedError

        decode = prefill

    if backend is None:
        backend = Backend(att_config=None, runtime_config=AttRuntimeConfig(logical_layer_ids=(0,)))
    runtime = _runtime(backend)
    owner.att_states = create_att_state(backend=runtime, infer_state=owner)
    return owner


@pytest.mark.parametrize("is_prefill", [False, True])
def test_att_state_do_not_keep_their_infer_state_alive(no_cycle_gc, is_prefill):
    owner = _create_state(is_prefill)
    (att_state,) = owner.att_states.values()
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
    owner.att_states = create_att_state(backend=_runtime(), infer_state=owner)
    assert owner.att_states == {}


@pytest.mark.parametrize(
    ("backend", "config", "field"),
    [
        (
            TritonGQAPrefillBackend,
            GQAAttConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16),
            "b_req_idx",
        ),
        (
            TritonMLAPrefillBackend,
            MLAPrefillAttConfig(
                num_heads=8,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                softmax_scale=0.125,
                dtype=torch.bfloat16,
            ),
            "b_seq_len",
        ),
        (
            FA3SparseMLAPrefillBackend,
            SparseMLAAttConfig(
                num_heads=8,
                kv_lora_rank=512,
                qk_rope_head_dim=0,
                softmax_scale=0.125,
                dtype=torch.bfloat16,
                kv_cache_format="fp8_group64_e8m0",
            ),
            "b_req_idx",
        ),
    ],
)
def test_prefill_backend_state_outlives_step(no_cycle_gc, backend, config, field):
    owner = InferStateInfo()
    owner.is_prefill = True
    owner.req_manager = SimpleNamespace(req_to_token_indexs=None)
    tensor = torch.arange(4)
    setattr(owner, field, tensor)
    impl = backend(config, runtime_config=AttRuntimeConfig(logical_layer_ids=(0,)))
    runtime = _runtime(impl)
    owner.att_states = create_att_state(backend=runtime, infer_state=owner)
    att_state = owner.att_states[impl]
    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None
    assert getattr(att_state, field) is tensor


def test_decode_graph_replay_preserves_capture_buffers_and_owner_lifetime(no_cycle_gc):
    captured = _create_state(is_prefill=False)
    captured.input_ids = torch.zeros(4, dtype=torch.int64)
    captured_ref = weakref.ref(captured)
    capture_positions_ptr = captured.position_ids.data_ptr()
    capture_derived_ptr = next(iter(captured.att_states.values())).derived.data_ptr()
    replay_calls = []

    graph = CudaGraph.__new__(CudaGraph)
    output = object()
    graph.graph = {4: (SimpleNamespace(replay=lambda: replay_calls.append(True)), captured, output)}
    del captured
    assert captured_ref() is not None

    backend = next(iter(captured_ref().att_states))
    runtime = _create_state(is_prefill=False, backend=backend)
    runtime.input_ids = torch.ones(4, dtype=torch.int64)
    runtime.position_ids.add_(7)
    runtime.att_states[backend].derived.copy_(runtime.position_ids * 2)
    source_derived_ref = weakref.ref(runtime.att_states[backend].derived)
    runtime_ref = weakref.ref(runtime)
    assert graph._replay(runtime) is output
    assert replay_calls == [True]
    # Replay copies the step's tensors into the captured state's, which its attention state read.
    assert captured_ref().position_ids.data_ptr() == capture_positions_ptr
    assert next(iter(captured_ref().att_states.values())).positions is captured_ref().position_ids
    torch.testing.assert_close(captured_ref().position_ids, runtime.position_ids)
    assert captured_ref().att_states[backend].derived.data_ptr() == capture_derived_ptr
    torch.testing.assert_close(captured_ref().att_states[backend].derived, runtime.att_states[backend].derived)

    del runtime
    assert runtime_ref() is None
    assert source_derived_ref() is None
    graph.graph.clear()
    assert captured_ref() is None
