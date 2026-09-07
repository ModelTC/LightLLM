from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import commit_sliding_window_state
from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow
from lightllm.common.kv_cache_mem_manager.operator.hybrid_sliding import HybridSlidingMemOperator
from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig, SlidingWindowStateCacheManager
from lightllm.models.gemma4.kv_layout import get_kv_cache_layout
from lightllm.models.gemma4.layer_infer.transformer_layer_infer import Gemma4TransformerLayerInfer
from lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm import context_attention_fwd_gemma4_mm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_request_window_view_shares_storage_and_has_no_second_token_table(monkeypatch):
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.ReqSamplingParamsManager", lambda size: None)
    config = SlidingWindowCacheConfig({0: 0, 2: 1}, {1: 0}, 32, 1, 64, 1, 64, torch.bfloat16)
    manager = ReqManagerForSlidingWindow(2, 128, None, config, scratch_token_num=7)
    assert manager.req_to_sliding_window.shape == (2, 3, 32, 2, 64)
    assert manager.req_to_sliding_window.data_ptr() == manager.sliding_kv_buffer.data_ptr()
    assert manager.req_to_sliding_window.stride(0) == manager.sliding_kv_buffer.stride(0)
    assert not hasattr(manager, "req_to_sliding_window_indexs")
    assert manager.req_to_token_indexs.shape == (3, 128)
    manager.req_to_sliding_window[:, 1].fill_(7)
    torch.testing.assert_close(
        manager.sliding_kv_buffer[:, 32:64], torch.full_like(manager.sliding_kv_buffer[:, 32:64], 7)
    )
    manager.sliding_kv_buffer[:, manager.scratch_start :].fill_(11)
    pages = SlidingWindowStateCacheManager(1, config)
    manager.save_small_page_state(1, 0, pages)
    manager.init_hybrid_attention_state(SimpleNamespace(req_idx=1))
    assert torch.count_nonzero(manager.req_to_sliding_window).item() == 0
    manager._restore_state(2, pages, 0)  # Include the reserved hold request slot.
    assert torch.all(manager.req_to_sliding_window[:, 2] == 7)
    assert torch.all(manager.sliding_kv_buffer[:, manager.scratch_start :] == 11)
    state = SimpleNamespace(input_ids=torch.zeros(7, dtype=torch.int32, device="cuda"))
    manager.prepare_sliding_window(state)
    assert torch.count_nonzero(manager.req_to_token_indexs).item() == 0
    torch.testing.assert_close(
        state.sliding_window_mem_index,
        torch.arange(manager.scratch_start, manager.scratch_start + 7, device="cuda"),
    )


@pytest.mark.parametrize("layer_index,is_shared", [(5, False), (11, False), (17, True)])
def test_full_kv_write_maps_logical_layer_once_and_skips_shared_readers(layer_index, is_shared):
    config = SlidingWindowCacheConfig({0: 0}, {5: 0, 11: 1, 17: 1}, 32, 1, 64, 1, 64, torch.bfloat16)
    mem_manager = SimpleNamespace(
        sliding_config=config, kv_buffer=torch.zeros((2, 8, 2, 64), dtype=torch.bfloat16, device="cuda")
    )
    mem_manager.operator = HybridSlidingMemOperator(mem_manager)
    layer = object.__new__(Gemma4TransformerLayerInfer)
    layer.layer_num_, layer.is_sliding, layer.is_kv_shared_ = layer_index, False, is_shared
    indexes = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    kv = torch.randn((2, 2, 64), dtype=torch.bfloat16, device="cuda")
    layer._post_cache_kv(kv, SimpleNamespace(mem_manager=mem_manager, mem_index=indexes), None)
    expected = torch.zeros_like(mem_manager.kv_buffer)
    if not is_shared:
        expected[config.get_full_layer_index(layer_index), indexes] = kv
    torch.testing.assert_close(mem_manager.kv_buffer, expected, atol=0, rtol=0)


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
    b_req = torch.tensor([req_idx], device="cuda", dtype=torch.int32)
    b_seq = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    b_q = torch.tensor([q_len], device="cuda", dtype=torch.int32)
    b_start = torch.tensor([0], device="cuda", dtype=torch.int32)
    b_history = torch.tensor([history_len], device="cuda", dtype=torch.int32)
    image_end = torch.zeros(q_len, device="cuda", dtype=torch.int32)
    q = torch.randn((q_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    actual, expected = torch.empty_like(q), torch.empty_like(q)
    reference_mapping = torch.arange(seq_len, device="cuda", dtype=torch.int32).expand(3, -1)
    for buffer, indexes, output, scratch in [
        (runtime, None, actual, scratch_start),
        (reference, reference_mapping, expected, None),
    ]:
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
            scratch_start=scratch,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    commit_sliding_window_state(runtime, b_req, b_seq, b_q, b_start, window, scratch_start, q_len)
    positions = torch.arange(max(0, seq_len - window), seq_len, device="cuda")
    torch.testing.assert_close(runtime[req_idx * window + positions % window], reference[positions], atol=0, rtol=0)
    assert torch.count_nonzero(runtime[:window]).item() == 0


@pytest.mark.parametrize("is_prefill", [True, False])
def test_shared_kv_is_committed_only_after_last_reader_and_snapshot_is_independent(is_prefill):
    window, history_len, q_len, head_dim = 512, 512, 256 if is_prefill else 1, 64
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
    manager.sliding_kv_buffer = torch.zeros((1, 2 * window + q_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    manager.req_to_sliding_window = manager.sliding_kv_buffer[:, : manager.scratch_start].view(
        1, 2, window, 2, head_dim
    )
    manager.sliding_kv_buffer[0, manager.scratch_start :, 1] = 1
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
        layer.alloc_tensor = lambda shape, dtype, device="cuda": torch.empty(shape, dtype=dtype, device=device)
        if is_prefill:
            outputs.append(layer._context_attention_kernel(q, None, state, None))
        else:
            outputs.append(layer._token_attention_kernel(q, state, None))
        if index == 0:
            assert torch.count_nonzero(manager.req_to_sliding_window[:, 0]).item() == 0
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert outputs[1][0, 0, 0].item() == 1 / window
    assert manager.req_to_sliding_window[0, 0, 0, 1, 0].item() == 1

    pages = SlidingWindowStateCacheManager(2, config)
    page = pages.alloc_one_state_cache()
    manager.save_small_page_state(0, page, pages)
    saved = pages.get_state_cache(page).clone()
    manager.req_to_sliding_window[:, 0].fill_(7)
    torch.testing.assert_close(pages.get_state_cache(page), saved, atol=0, rtol=0)
    manager._restore_state(1, pages, page)
    torch.testing.assert_close(manager.req_to_sliding_window[:, 1], saved, atol=0, rtol=0)
    pages.free_state_cache([page])
    assert pages.get_free_cache_num() == 2


def test_empty_snapshot_does_not_read_gpu_request_ids():
    manager = object.__new__(ReqManagerForSlidingWindow)
    # No runtime or page pool is needed for a no-op. In particular, no .tolist()
    # or other GPU operation should be performed on b_req_idx.
    manager.save_big_page_states(object(), [0, 1], [-1, -1])


@pytest.mark.parametrize(
    "window,q_lengths",
    [(32, [6, 5, 32]), (512, [4096, 1, 513]), (512, [1, 8192, 511]), (1024, [8192, 4096, 1])],
)
def test_batched_window_commit_with_hold_request_and_cuda_graph_replay(window, q_lengths):
    scratch_start, head_dim = 4 * window, 64
    req_ids = [2, 0, 3]
    lengths = [q_lengths[0] + 2 * window + 3, q_lengths[1], q_lengths[2] + window - 1]
    replay_lengths = [length + window + 7 for length in lengths]
    starts = [0, q_lengths[0], q_lengths[0] + q_lengths[1]]
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    b_req, b_seq, b_q, b_start = map(int_tensor, [req_ids, lengths, q_lengths, starts])
    runtime = torch.zeros((scratch_start + sum(q_lengths), 2, head_dim), device="cuda", dtype=torch.bfloat16)
    runtime[scratch_start:] = torch.randn_like(runtime[scratch_start:])

    def forward():
        commit_sliding_window_state(runtime, b_req, b_seq, b_q, b_start, window, scratch_start, max(q_lengths))

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    # Replay with changed GPU state, including a padded/hold request ID.
    runtime[:scratch_start].zero_()
    runtime[scratch_start:].mul_(2)
    b_seq.copy_(int_tensor(replay_lengths))
    graph.replay()
    for req, seq, q_len, start in zip(req_ids, replay_lengths, q_lengths, starts):
        tail_start = max(q_len - window, 0)
        positions = torch.arange(seq - q_len + tail_start, seq, device="cuda")
        expected_ring = torch.zeros_like(runtime[req * window : (req + 1) * window])
        expected_ring[positions % window] = runtime[scratch_start + start + tail_start : scratch_start + start + q_len]
        torch.testing.assert_close(runtime[req * window : (req + 1) * window], expected_ring, atol=0, rtol=0)
    assert torch.count_nonzero(runtime[window : 2 * window]).item() == 0


@pytest.mark.parametrize("max_q_seq_len", [1, 511, 512, 513, 4096, 8192])
def test_commit_grid_is_bounded_by_window(monkeypatch, max_q_seq_len):
    import lightllm.common.basemodel.triton_kernel.sliding_window_state as state_kernel

    grids = []

    class RecordingKernel:
        def __getitem__(self, grid):
            grids.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(state_kernel, "_commit_sliding_window_state", RecordingKernel())
    layer_buffer = SimpleNamespace(shape=(8192, 2, 64), stride=lambda: (128, 64, 1))
    req_ids = SimpleNamespace(shape=(3,))
    state_kernel.commit_sliding_window_state(layer_buffer, req_ids, None, None, None, 512, 2048, max_q_seq_len)
    assert grids == [(3, min(max_q_seq_len, 512), 2)]
