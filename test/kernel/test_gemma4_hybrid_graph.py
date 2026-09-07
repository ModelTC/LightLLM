from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.triton.fp import TritonDecodeAttState
from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow
from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig
from lightllm.models.gemma4.infer_struct import Gemma4InferStateInfo
from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _int_tensor(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _layer(is_sliding, shared):
    layer = object.__new__(Gemma4TransformerLayerInfer)
    layer.is_sliding, layer.is_kv_shared_ = is_sliding, shared
    layer.layer_num_ = (4 if shared else 2) if is_sliding else (3 if shared else 1)
    layer.kv_share_target_layer_ = (2 if is_sliding else 1) if shared else None
    layer.tp_q_head_num_, layer.head_dim_ = (4, 256) if is_sliding else (8, 512)
    layer.commit_sliding_state_ = False  # Isolate graph metadata and KV reads from snapshot/commit tests.
    layer.alloc_tensor = lambda shape, dtype, device="cuda": torch.empty(shape, dtype=dtype, device=device)
    return layer


def _state(model, req_manager, req_ids, seq_lengths, q_starts):
    state = Gemma4InferStateInfo()
    state.req_manager, state.mem_manager = req_manager, req_manager.mem_manager
    state.is_prefill = False
    state.batch_size, state.max_q_seq_len = len(req_ids), 1
    state.max_kv_seq_len = req_manager.req_to_token_indexs.shape[1]
    state.b_req_idx, state.b_seq_len = _int_tensor(req_ids), _int_tensor(seq_lengths)
    state.input_ids = torch.zeros(len(req_ids), device="cuda", dtype=torch.int64)
    backend = SimpleNamespace(model=model)
    state.decode_att_state = TritonDecodeAttState(backend=backend, infer_state=state)
    state.decode_att_state1 = TritonDecodeAttState(backend=backend, infer_state=state)
    state.init_some_extra_state(model)
    # Independently vary scratch locations to expose stale graph metadata.
    state.b_q_start_loc = _int_tensor(q_starts)
    state.init_att_state()
    return state


@pytest.mark.parametrize("window", [512, 1024])
@pytest.mark.parametrize("shared", [False, True])
def test_gemma_hybrid_decode_graph_copies_state_without_replacing_full_token_table(monkeypatch, window, shared):
    from lightllm.common.triton_utils import autotuner

    monkeypatch.setattr(autotuner, "get_triton_autotune_level", lambda: autotuner.AutotuneLevel.CLOSE_AUTOTUNE)
    torch.manual_seed(42)
    req_slots, batch_size, max_seq_len = 6, 4, 3 * window + 32
    config = SlidingWindowCacheConfig({0: 0, 2: 1, 4: 1}, {1: 0, 3: 0}, window, 1, 256, 2, 512, torch.bfloat16)
    # Skip launch-time distributed/profile setup, retaining the real cache access methods.
    mem_manager = object.__new__(HybridSlidingMemoryManager)
    mem_manager.sliding_config, mem_manager.head_num = config, config.full_head_num
    mem_manager.kv_buffer = torch.randn((1, 1024, 4, 512), device="cuda", dtype=config.dtype)
    manager = object.__new__(ReqManagerForSlidingWindow)
    manager.mem_manager, manager.sliding_config = mem_manager, config
    manager.sliding_window, manager.scratch_token_num = window, batch_size
    manager.scratch_start = req_slots * window
    manager.sliding_kv_buffer = torch.randn(
        (2, manager.scratch_start + batch_size, 2, 256), device="cuda", dtype=config.dtype
    )
    manager.req_to_sliding_window = manager.sliding_kv_buffer[:, : manager.scratch_start].view(
        2, req_slots, window, 2, 256
    )
    manager.req_to_token_indexs = torch.randint(1024, (req_slots, max_seq_len), device="cuda", dtype=torch.int32)
    original_table = manager.req_to_token_indexs
    expected_table = original_table.clone()
    cos = torch.ones((max_seq_len, 128), device="cuda", dtype=config.dtype)
    sin = torch.zeros_like(cos)
    model = SimpleNamespace(
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda is_draft: 0),
        is_mtp_draft_model=False,
        _cos_cached_sliding=cos,
        _sin_cached_sliding=sin,
        _cos_cached_full=cos,
        _sin_cached_full=sin,
    )
    captured = _state(model, manager, [0, 2, 5, 5], [window + 5, 17, 2, 2], [0, 1, 2, 3])
    captured.is_cuda_graph = True
    sliding, full = _layer(True, shared), _layer(False, shared)
    sliding.sliding_window_, full.sliding_window_ = window, 0
    q_sliding = torch.randn((batch_size, 4, 256), device="cuda", dtype=config.dtype)
    q_full = torch.randn((batch_size, 8, 512), device="cuda", dtype=config.dtype)

    def forward(state):
        return (
            sliding._token_attention_kernel(q_sliding, state, None),
            full._token_attention_kernel(q_full, state, None),
        )

    forward(captured)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = forward(captured)
    copied_fields = ("b_req_idx", "b_seq_len", "b_q_start_loc", "position_ids", "sliding_window_mem_index")
    captured_ptrs = {name: getattr(captured, name).data_ptr() for name in copied_fields}
    new_state = _state(model, manager, [3, 1, 5, 5], [2 * window + 11, 1, 2, 2], [2, 0, 1, 3])
    captured.copy_for_cuda_graph(new_state)
    q_sliding.mul_(0.5)
    q_full.mul_(0.75)
    manager.sliding_kv_buffer.mul_(0.75)
    mem_manager.kv_buffer.mul_(0.5)
    graph.replay()
    eager_outputs = forward(new_state)

    for actual, expected in zip(graph_outputs, eager_outputs):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.isfinite(actual).all()
    for name in copied_fields:
        assert getattr(captured, name).data_ptr() == captured_ptrs[name]
        torch.testing.assert_close(getattr(captured, name), getattr(new_state, name), atol=0, rtol=0)
    for state in (captured, new_state):
        assert state.decode_att_state.infer_state is state
        assert state.decode_att_state1.infer_state is state
        assert state.req_manager is manager
        assert state.req_manager.req_to_token_indexs is original_table
    assert not hasattr(manager, "req_to_sliding_window_indexs")
    torch.testing.assert_close(original_table, expected_table, atol=0, rtol=0)
