import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import InferStateInfo
from lightllm.core.runner import batch_metadata as batch_metadata
from lightllm.core.runner.graph import GraphRunner
from lightllm.models.gemma3.infer_struct import Gemma3InferStateInfo
from lightllm.models.gemma4.infer_struct import Gemma4InferStateInfo
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.neo_chat_moe.infer_struct import NeoChatInferStateInfo
from lightllm.models.qwen.infer_struct import QwenInferStateInfo
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo


@pytest.fixture(params=["cpu", "cuda"])
def device(request, monkeypatch):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        return "cuda"

    def prepare(state):
        lengths = state.b_seq_len.tolist()
        ready = state.b_ready_cache_len.tolist() if state.is_prefill else [length - 1 for length in lengths]
        state.position_ids = torch.tensor([pos for start, end in zip(ready, lengths) for pos in range(start, end)])
        state.b_q_seq_len = torch.tensor([end - start for start, end in zip(ready, lengths)])
        state.b1_cu_q_seq_len = torch.cat((torch.tensor([0]), state.b_q_seq_len.cumsum(0)))
        state.b_q_start_loc = state.b1_cu_q_seq_len[:-1]

    monkeypatch.setattr(batch_metadata, "prepare_sequence_metadata", prepare)
    return "cpu"


@pytest.mark.parametrize("prefill", [True, False])
@pytest.mark.parametrize("kind", ["nope", "llama", "gemma3", "gemma4", "qwen", "qwen_ntk", "qwen_vl", "neo"])
def test_bound_metadata_keeps_native_tables_without_model(device, prefill, kind):
    classes = {
        "nope": InferStateInfo,
        "llama": LlamaInferStateInfo,
        "gemma3": Gemma3InferStateInfo,
        "gemma4": Gemma4InferStateInfo,
        "qwen": QwenInferStateInfo,
        "qwen_ntk": QwenInferStateInfo,
        "qwen_vl": Qwen2VLInferStateInfo,
        "neo": NeoChatInferStateInfo,
    }
    model = type("MetadataSource", (), {})()
    model.config = {"use_dynamic_ntk": kind == "qwen_ntk", "seq_length": 4, "rope_scaling": {"type": "mrope"}}
    model.max_ntk_alpha = 2
    model.logn_tensor = torch.arange(32, dtype=torch.float32, device=device) / 3
    table = torch.arange(64, dtype=torch.float32, device=device).view(32, 2)
    if kind == "qwen_ntk":
        table = torch.stack((table, table + 100, table + 200))
    for suffix, offset in (("", 0), ("_local", 10), ("_global", 20), ("_sliding", 30), ("_full", 40)):
        setattr(model, "_cos_cached" + suffix, table + offset)
        setattr(model, "_sin_cached" + suffix, -table - offset)
    model._hw_cos_cached, model._hw_sin_cached = table + 50, -table - 50
    prepare = classes[kind].bind_metadata(model)
    model_ref = weakref.ref(model)
    model.config.clear()
    model._cos_cached = model._sin_cached = None
    del model
    assert model_ref() is None

    def make_state():
        state = classes[kind]()
        state.is_prefill = prefill
        state.batch_size = 2
        state.max_kv_seq_len = 7
        state.input_ids = torch.zeros(5 if prefill else 2, dtype=torch.int64, device=device)
        state.b_seq_len = torch.tensor([4, 7], dtype=torch.int32, device=device)
        state.b_ready_cache_len = torch.tensor([2, 4], dtype=torch.int32, device=device)
        state.multimodal_params = []
        if kind == "qwen_vl" and not prefill:
            state.b_position_delta = torch.tensor([1, -2], dtype=torch.int32, device=device)
        if kind == "neo" and not prefill:
            state.multimodal_params = [{"images": [{"grid_thwd": [1, 1, 1, delta]}]} for delta in (1, -2)]
        prepare(state)
        return state

    state = make_state()
    positions = torch.tensor([2, 3, 4, 5, 6] if prefill else [3, 6], device=device)
    expected_positions = positions
    if kind == "qwen_vl":
        expected_positions = (positions if prefill else positions + positions.new_tensor([1, -2])).expand(3, -1)
        assert state.rope_type == "mrope"
    elif kind == "neo":
        expected_positions = torch.stack((positions, positions * 0, positions * 0))
        if not prefill:
            expected_positions[0] += positions.new_tensor([1, -2])
    torch.testing.assert_close(state.position_ids, expected_positions, check_dtype=False)
    if kind == "nope":
        return
    if kind == "qwen_ntk":
        levels = positions.new_tensor([1, 1, 2, 2, 2] if prefill else [1, 2])
        expected_cos = table[levels, positions]
        torch.testing.assert_close(state.logn_values, positions.float() / 3)
    else:
        expected_cos = table[expected_positions[0] if kind == "neo" else expected_positions]
    suffixes = {"gemma3": (("", 0), ("_local", 10), ("_global", 20)), "gemma4": (("_sliding", 30), ("_full", 40))}.get(
        kind, (("", 0),)
    )
    for suffix, offset in suffixes:
        torch.testing.assert_close(getattr(state, "position_cos" + suffix), expected_cos + offset)
        torch.testing.assert_close(getattr(state, "position_sin" + suffix), -expected_cos - offset)
    if kind == "neo":
        for axis, suffix in ((1, "_h"), (2, "_w")):
            torch.testing.assert_close(getattr(state, "position_cos" + suffix), table[expected_positions[axis]] + 50)
    second = make_state()
    for suffix, _ in suffixes:
        getattr(second, "position_cos" + suffix).zero_()
        torch.testing.assert_close(getattr(state, "position_cos" + suffix), expected_cos + dict(suffixes)[suffix])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gemma4_image_visibility_keeps_prefix_offsets():
    table = torch.zeros((16, 4), device="cuda")
    prepare = Gemma4InferStateInfo.bind_metadata(
        SimpleNamespace(
            _cos_cached_sliding=table,
            _sin_cached_sliding=table,
            _cos_cached_full=table,
            _sin_cached_full=table,
        )
    )
    state = Gemma4InferStateInfo()
    state.is_prefill = True
    state.max_kv_seq_len = 7
    state.input_ids = torch.zeros(5, dtype=torch.int64, device="cuda")
    state.b_seq_len = torch.tensor([4, 7], dtype=torch.int32, device="cuda")
    state.b_ready_cache_len = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
    state.multimodal_params = [{"images": [{"start_idx": start, "token_num": 2}]} for start in (2, 5)]
    prepare(state)
    assert state.b_image_token_end.tolist() == [4, 4, 0, 7, 7]


def test_metadata_capacity_observes_graph_installed_after_binding():
    runner = GraphRunner()
    capacity = runner.decode_metadata_capacity
    assert capacity(2, 258) == 258
    runner.decode = SimpleNamespace(
        can_run=lambda batch, length: batch <= 4 and length <= 16384, graph_max_len_in_batch=16384
    )
    assert capacity(2, 258) == 16384
    assert capacity(5, 258) == 258
    assert capacity(2, 16385) == 16385
