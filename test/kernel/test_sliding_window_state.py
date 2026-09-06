from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import (
    commit_sliding_window_state,
    prepare_sliding_window_indexes,
)
from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow
from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig, SlidingWindowStateCacheManager
from lightllm.models.gemma4.kv_layout import get_kv_cache_layout
from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer
from lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm import context_attention_fwd_gemma4_mm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("history_len", [0, 511, 512, 513, 1024])
@pytest.mark.parametrize("q_len", [1, 31, 256, 768])
def test_ring_attention_and_commit_match_token_cache(history_len, q_len):
    torch.manual_seed(42)
    window, head_dim, req_idx = 512, 64, 1
    seq_len = history_len + q_len
    scratch_start = 3 * window
    reference = torch.randn((seq_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    runtime = torch.zeros((scratch_start + q_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    old_positions = torch.arange(max(0, history_len - window), history_len, device="cuda")
    runtime[req_idx * window + old_positions % window] = reference[old_positions]
    runtime[scratch_start:] = reference[history_len:]
    mapping = torch.full((3, seq_len), -1, device="cuda", dtype=torch.int32)
    b_req = torch.tensor([req_idx], device="cuda", dtype=torch.int32)
    b_seq = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    b_q = torch.tensor([q_len], device="cuda", dtype=torch.int32)
    b_start = torch.tensor([0], device="cuda", dtype=torch.int32)
    b_history = torch.tensor([history_len], device="cuda", dtype=torch.int32)
    image_end = torch.zeros(q_len, device="cuda", dtype=torch.int32)
    prepare_sliding_window_indexes(mapping, b_req, b_seq, b_q, b_start, window, scratch_start, q_len)
    q = torch.randn((q_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    actual, expected = torch.empty_like(q), torch.empty_like(q)
    reference_mapping = torch.arange(seq_len, device="cuda", dtype=torch.int32).expand(3, -1)
    for buffer, indexes, output in [(runtime, mapping, actual), (reference, reference_mapping, expected)]:
        context_attention_fwd_gemma4_mm(
            q,
            buffer[:, :1],
            buffer[:, 1:],
            output,
            b_req,
            b_start,
            b_seq,
            b_history,
            q_len,
            indexes,
            image_end,
            sliding_window=(window - 1, 0),
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    commit_sliding_window_state(runtime, b_req, b_seq, b_q, b_start, window, scratch_start, q_len)
    positions = torch.arange(max(0, seq_len - window), seq_len, device="cuda")
    torch.testing.assert_close(runtime[req_idx * window + positions % window], reference[positions], atol=0, rtol=0)
    assert torch.count_nonzero(runtime[:window]).item() == 0


def test_shared_kv_is_committed_only_after_last_reader_and_snapshot_is_independent():
    window, history_len, q_len, head_dim = 512, 512, 256, 64
    layout, owners, last_reader = get_kv_cache_layout(
        {"layer_types": ["sliding_attention", "full_attention", "sliding_attention"], "num_kv_shared_layers": 1}
    )
    config = SlidingWindowCacheConfig(
        layout["sliding_attention"], layout["full_attention"], window, 1, head_dim, 1, 64, torch.bfloat16
    )
    manager = object.__new__(ReqManagerForSlidingWindow)
    manager.sliding_config = config
    manager.sliding_window = window
    manager.scratch_token_num = q_len
    manager.scratch_start = 2 * window
    manager.req_to_sliding_window = torch.zeros(
        (1, 2 * window + q_len, 2, head_dim), device="cuda", dtype=torch.bfloat16
    )
    manager.req_to_sliding_window[0, manager.scratch_start :, 1] = 1
    manager.req_to_sliding_window_indexs = torch.zeros((2, history_len + q_len), device="cuda", dtype=torch.int32)
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    state = SimpleNamespace(
        input_ids=int_tensor([0] * q_len),
        b_req_idx=int_tensor([0]),
        b_seq_len=int_tensor([history_len + q_len]),
        b_q_seq_len=int_tensor([q_len]),
        b_q_start_loc=int_tensor([0]),
        b_ready_cache_len=int_tensor([history_len]),
        max_q_seq_len=q_len,
        b_image_token_end=int_tensor([0] * q_len),
        req_manager=manager,
    )
    manager.prepare_sliding_window(state)
    q = torch.zeros((q_len, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    outputs = []
    for index in [0, 2]:
        layer = object.__new__(Gemma4TransformerLayerInfer)
        layer.layer_num_, layer.is_sliding, layer.sliding_window_ = index, True, window
        layer.is_kv_shared_, layer.kv_share_target_layer_ = index != owners[index], owners[index]
        layer.commit_sliding_state_ = last_reader[owners[index]] == index
        layer.tp_q_head_num_, layer.head_dim_ = 1, head_dim
        layer.alloc_tensor = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
        outputs.append(layer._context_attention_kernel(q, None, state, None))
        if index == 0:
            assert torch.count_nonzero(manager.req_to_sliding_window[:, :window]).item() == 0
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert outputs[1][0, 0, 0].item() == 1 / window
    assert manager.req_to_sliding_window[0, 0, 1, 0].item() == 1

    pages = SlidingWindowStateCacheManager(2, config)
    page = pages.alloc_one_state_cache()
    manager.save_small_page_state(0, page, pages)
    saved = pages.get_state_cache(page).clone()
    manager.req_to_sliding_window[:, :window].fill_(7)
    torch.testing.assert_close(pages.get_state_cache(page), saved, atol=0, rtol=0)
    manager._restore_state(1, pages, page)
    torch.testing.assert_close(manager.req_to_sliding_window[:, window : 2 * window], saved, atol=0, rtol=0)
    pages.free_state_cache([page])
    assert pages.get_free_cache_num() == 2


def test_empty_snapshot_does_not_read_gpu_request_ids():
    manager = object.__new__(ReqManagerForSlidingWindow)
    # No runtime or page pool is needed for a no-op. In particular, no .tolist()
    # or other GPU operation should be performed on b_req_idx.
    manager.save_big_page_states(object(), [0, 1], [-1, -1])


def test_batched_window_commit_with_hold_request_and_cuda_graph_replay():
    window, scratch_start, head_dim = 32, 4 * 32, 64
    req_ids, lengths, q_lengths = [2, 0, 3], [86, 5, 100], [6, 5, 32]
    starts = [0, 6, 11]
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    b_req, b_seq, b_q, b_start = map(int_tensor, [req_ids, lengths, q_lengths, starts])
    mapping = torch.full((4, 128), -1, device="cuda", dtype=torch.int32)
    runtime = torch.zeros((scratch_start + sum(q_lengths), 2, head_dim), device="cuda", dtype=torch.bfloat16)
    runtime[scratch_start:] = torch.randn_like(runtime[scratch_start:])

    def forward():
        prepare_sliding_window_indexes(mapping, b_req, b_seq, b_q, b_start, window, scratch_start, max(q_lengths))
        commit_sliding_window_state(runtime, b_req, b_seq, b_q, b_start, window, scratch_start, max(q_lengths))

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    # Replay with changed GPU state, including a padded/hold request ID.
    runtime[:scratch_start].zero_()
    runtime[scratch_start:].mul_(2)
    graph.replay()
    for req, seq, q_len, start in zip(req_ids, lengths, q_lengths, starts):
        positions = torch.arange(seq - q_len, seq, device="cuda")
        expected = runtime[scratch_start + start : scratch_start + start + q_len]
        torch.testing.assert_close(runtime[req * window + positions % window], expected, atol=0, rtol=0)
        torch.testing.assert_close(
            mapping[req, positions],
            torch.arange(scratch_start + start, scratch_start + start + q_len, device="cuda", dtype=torch.int32),
        )
    assert torch.count_nonzero(runtime[window : 2 * window]).item() == 0
