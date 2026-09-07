import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import (
    get_sliding_window_mem_indexes,
    move_sliding_window,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("window,q_lengths", [(32, [1, 31, 65]), (512, [4096, 1, 513]), (1024, [8192, 7, 1023])])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefill_moves_all_layers_and_preserves_canonical_snapshot(window, q_lengths, dtype):
    torch.manual_seed(42)
    req_ids, histories = [3, 0, 5], [0, window - 1, 2 * window + 3]
    starts = [11, 18 + q_lengths[0], 31 + q_lengths[0] + q_lengths[1]]
    lengths = [history + q_len for history, q_len in zip(histories, q_lengths)]
    total_tokens = starts[-1] + q_lengths[-1] + 7
    runtime_start = 6 * window + 17
    pool = torch.full((3, runtime_start + total_tokens + 3 * window, 4, 32), -11, device="cuda", dtype=dtype)
    references = [torch.randn((3, length, 4, 32), device="cuda", dtype=dtype) for length in lengths]
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    b_req, b_seq, b_history, b_q, b_start = map(int_tensor, [req_ids, lengths, histories, q_lengths, starts])
    for req, history, reference in zip(req_ids, histories, references):
        positions = torch.arange(max(0, history - window), history, device="cuda")
        pool[:, req * window + positions % window] = reference[:, positions]

    expected = pool.clone()
    for batch, (history, start, reference) in enumerate(zip(histories, starts, references)):
        positions = torch.arange(max(0, history - window), history, device="cuda")
        current_start = runtime_start + start + (batch + 1) * window
        expected[:, current_start + positions - history] = reference[:, positions]
    move_sliding_window(pool, b_req, b_seq, b_history, b_start, window, runtime_start)
    torch.testing.assert_close(pool, expected, atol=0, rtol=0)

    indexes = get_sliding_window_mem_indexes(
        b_req, b_seq, b_q, b_start, window, runtime_start, total_tokens, max(q_lengths), is_prefill=True
    )
    for batch, (start, q_len, history, reference) in enumerate(zip(starts, q_lengths, histories, references)):
        expected_indexes = runtime_start + start + (batch + 1) * window + torch.arange(q_len, device="cuda")
        torch.testing.assert_close(indexes[start : start + q_len].long(), expected_indexes, atol=0, rtol=0)
        pool[:, indexes[start : start + q_len].long()] = reference[:, history:]

    expected = pool.clone()
    for req, length, reference in zip(req_ids, lengths, references):
        positions = torch.arange(max(0, length - window), length, device="cuda")
        expected[:, req * window + positions % window] = reference[:, positions]
    move_sliding_window(pool, b_req, b_seq, b_history, b_start, window, runtime_start, compact=True)
    torch.testing.assert_close(pool, expected, atol=0, rtol=0)
    # The page format remains a raw W-slot ring, independent of the active area.
    snapshot = pool[:, 5 * window : 6 * window].clone()
    pool[:, runtime_start:].zero_()
    torch.testing.assert_close(snapshot, expected[:, 5 * window : 6 * window], atol=0, rtol=0)


@pytest.mark.parametrize("window", [32, 512, 1024])
def test_decode_indexes_cuda_graph_replay(window):
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    b_req, b_seq = int_tensor([2, 0, 5]), int_tensor([1, window, 2 * window + 1])

    def forward():
        return get_sliding_window_mem_indexes(b_req, b_seq, None, None, window, 6 * window, 3, 1, False)

    torch.testing.assert_close(forward(), int_tensor([2 * window, window - 1, 5 * window]), atol=0, rtol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexes = forward()
    b_req.copy_(int_tensor([5, 1, 3]))
    b_seq.copy_(int_tensor([window + 3, 7, 4 * window]))
    graph.replay()
    torch.testing.assert_close(indexes, int_tensor([5 * window + 2, window + 6, 4 * window - 1]), atol=0, rtol=0)
