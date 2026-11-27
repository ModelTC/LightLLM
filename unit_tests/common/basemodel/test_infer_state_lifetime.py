"""An InferStateInfo and its attention attention state are freed by reference counting, without cycle GC."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.attention_backend_manager import AttentionBackendManager
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.attention.backends.gqa import GQALayerConfig
from lightllm.common.attention.backends.gqa.triton import TritonGQAPrefillBackend
from lightllm.common.attention.backends.mla import MLAPrefillLayerConfig
from lightllm.common.attention.backends.mla.triton import TritonMLAPrefillBackend
from lightllm.common.attention.backends.sparse_mla import SparseMLALayerConfig
from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLAPrefillBackend


class _AttMetadata:
    def __init__(self, positions):
        self.positions = positions


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


def _runtime(groups: dict) -> AttentionBackendManager:
    """A manager whose main role has the given state groups per phase, without registering layers."""
    runtime = AttentionBackendManager()
    runtime.group_backends["main"] = {"prefill": {}, "decode": {}, **groups}
    return runtime


def _create_state(is_prefill):
    owner = InferStateInfo()
    owner.is_prefill = is_prefill
    owner.position_ids = torch.arange(4)
    owner.b_req_idx = owner.position_ids
    owner.req_manager = SimpleNamespace(req_to_token_indexs=None)
    backend = SimpleNamespace(create_att_metadata=lambda metadata: _AttMetadata(metadata.b_req_idx))
    phase = "prefill" if is_prefill else "decode"
    runtime = _runtime({phase: {"group": backend}})
    owner.att_metadata = runtime.create_att_metadata_by_group(owner)
    return owner


@pytest.mark.parametrize("is_prefill", [False, True])
def test_att_metadata_does_not_keep_their_infer_state_alive(no_cycle_gc, is_prefill):
    owner = _create_state(is_prefill)
    att_metadata = owner.att_metadata["group"]
    owner_ref = weakref.ref(owner)
    tensor_ref = weakref.ref(owner.position_ids)

    # An external reference to the attention state must not retain their InferStateInfo.
    del owner
    assert owner_ref() is None
    assert tensor_ref() is not None
    metadata_ref = weakref.ref(att_metadata)
    del att_metadata
    assert metadata_ref() is None
    assert tensor_ref() is None


def test_infer_state_without_attention_builds_no_att_metadata():
    owner = InferStateInfo()
    owner.is_prefill = False
    owner.att_metadata = _runtime({}).create_att_metadata_by_group(owner)
    assert owner.att_metadata == {}


@pytest.mark.parametrize(
    ("backend_cls", "config", "field"),
    [
        (TritonGQAPrefillBackend, GQALayerConfig(8, 2, 64, torch.bfloat16), "b_req_idx"),
        (TritonMLAPrefillBackend, MLAPrefillLayerConfig(8, 128, 64, 128, 0.125, torch.bfloat16), "b_seq_len"),
        (
            FA3SparseMLAPrefillBackend,
            SparseMLALayerConfig(8, 512, 0, 0.125, torch.bfloat16, kv_cache_format="fp8_group64_e8m0"),
            "b_req_idx",
        ),
    ],
)
def test_prefill_backend_state_outlives_step(no_cycle_gc, backend_cls, config, field):
    owner = InferStateInfo()
    owner.is_prefill = True
    owner.req_manager = SimpleNamespace(req_to_token_indexs=None)
    tensor = torch.arange(4)
    setattr(owner, field, tensor)
    backend = backend_cls(config)
    runtime = _runtime({"prefill": {backend.metadata_group_key: backend}})
    owner.att_metadata = runtime.create_att_metadata_by_group(owner)
    att_metadata = owner.att_metadata[backend.metadata_group_key]
    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None
    assert getattr(att_metadata, field) is tensor


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
    assert captured_ref().att_metadata["group"].positions is captured_ref().position_ids
    torch.testing.assert_close(captured_ref().position_ids, runtime.position_ids)

    del runtime
    assert runtime_ref() is None
    graph.graph.clear()
    assert captured_ref() is None
