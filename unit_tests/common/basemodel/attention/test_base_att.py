import importlib
import inspect
from dataclasses import dataclass

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import BaseAttBackend, BaseDecodeAttState

_BACKEND_MODULES = (
    "triton.fp",
    "triton.int4kv",
    "triton.int8kv",
    "triton.mla",
    "triton.neo",
    "nsa.glm5_next",
    "linear.gdn",
    "linear.kda",
    "linear.replay_gdn",
    "linear.replay_kda",
)


def _all_backend_classes(base=BaseAttBackend):
    for subclass in base.__subclasses__():
        yield subclass
        yield from _all_backend_classes(subclass)


def test_concrete_backends_implement_state_factories():
    for module_name in _BACKEND_MODULES:
        importlib.import_module(f"lightllm.common.basemodel.attention.{module_name}")
    backend_classes = set(_all_backend_classes())
    assert backend_classes, "no attention backend classes were imported"

    for backend_class in backend_classes:
        if not inspect.isabstract(backend_class):
            continue
        # Only intermediate bases may stay abstract, and never because of the state factories.
        missing = backend_class.__abstractmethods__ & {"create_att_prefill_state", "create_att_decode_state"}
        assert not missing, f"{backend_class.__name__} does not implement {sorted(missing)}"


def test_backend_without_state_factories_cannot_be_instantiated():
    class IncompleteAttBackend(BaseAttBackend):
        def create_att_prefill_state(self, infer_state):
            return None

    with pytest.raises(TypeError, match="create_att_decode_state"):
        IncompleteAttBackend(model=object())


def test_default_decode_cuda_graph_copy_refreshes_tensor_fields_in_place():
    @dataclass
    class _DecodeState(BaseDecodeAttState):
        b_kv_seq_len: torch.Tensor = None
        shared: torch.Tensor = None
        label: str = ""

        def init_state(self):
            pass

        def decode_att(self, q, k, v, att_control=None, alloc_func=torch.empty):
            return q

    shared = torch.tensor([7, 7])
    graph_state = _DecodeState(b_kv_seq_len=torch.tensor([1, 2]), shared=shared, label="graph")
    graph_buffer = graph_state.b_kv_seq_len
    new_state = _DecodeState(b_kv_seq_len=torch.tensor([5, 6]), shared=shared, label="new")

    graph_state.copy_for_decode_cuda_graph(new_state)

    # The captured graph holds the original buffer, so values must be copied into it, not rebound.
    assert graph_state.b_kv_seq_len is graph_buffer
    assert graph_buffer.tolist() == [5, 6]
    assert graph_state.shared is shared
    assert graph_state.label == "graph"
