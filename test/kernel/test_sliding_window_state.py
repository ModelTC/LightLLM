import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_state import (
    get_sliding_window_decode_indexes,
    commit_sliding_window_kv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "window,q_lengths", [(32, [1, 31, 65]), (128, [1, 127, 513]), (512, [4096, 1, 513]), (1024, [8192, 7, 1023])]
)
@pytest.mark.parametrize(
    "dtype,payload_shape",
    [(torch.bfloat16, (4, 32)), (torch.float16, (4, 32)), (torch.bfloat16, (512,)), (torch.uint8, (584,))],
)
def test_prefill_commits_all_layers_to_fixed_rings(window, q_lengths, dtype, payload_shape):
    torch.manual_seed(42)
    req_ids, histories = [3, 0, 5], [0, window - 1, 2 * window + 3]
    starts = [11, 18 + q_lengths[0], 31 + q_lengths[0] + q_lengths[1]]
    lengths = [history + q_len for history, q_len in zip(histories, q_lengths)]
    total_tokens = starts[-1] + q_lengths[-1] + 7
    prefill_start = 6 * window
    pool = torch.full((3, prefill_start + total_tokens, *payload_shape), 11, device="cuda", dtype=dtype)
    state = pool[:, :prefill_start].unflatten(1, (6, window))
    references = [
        torch.randint(0, 256, (3, length, *payload_shape), device="cuda", dtype=dtype)
        if dtype == torch.uint8
        else torch.randn((3, length, *payload_shape), device="cuda", dtype=dtype)
        for length in lengths
    ]
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    b_req, b_seq, b_history, b_start = map(int_tensor, [req_ids, lengths, histories, starts])
    for req, history, reference in zip(req_ids, histories, references):
        positions = torch.arange(max(0, history - window), history, device="cuda")
        state[:, req, positions % window] = reference[:, positions]

    indexes = torch.arange(prefill_start, prefill_start + total_tokens, device="cuda", dtype=torch.int32)
    for start, q_len, history, reference in zip(starts, q_lengths, histories, references):
        pool[:, indexes[start : start + q_len].long()] = reference[:, history:]
    expected = pool.clone()
    for req, length, reference in zip(req_ids, lengths, references):
        positions = torch.arange(max(0, length - window), length, device="cuda")
        expected[:, req * window + positions % window] = reference[:, positions]
    commit_sliding_window_kv(pool, indexes, b_req, b_seq, b_history, b_start, window)
    # Check every layer, untouched requests, query gaps and the temporary region.
    torch.testing.assert_close(pool, expected, atol=0, rtol=0)


@pytest.mark.parametrize("window", [32, 512, 1024])
def test_decode_indexes_cuda_graph_replay(window):
    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    b_req, b_seq = int_tensor([2, 0, 5]), int_tensor([1, window, 2 * window + 1])

    def forward():
        return get_sliding_window_decode_indexes(b_req, b_seq, window)

    torch.testing.assert_close(forward(), int_tensor([2 * window, window - 1, 5 * window]), atol=0, rtol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexes = forward()
    b_req.copy_(int_tensor([5, 1, 3]))
    b_seq.copy_(int_tensor([window + 3, 7, 4 * window]))
    graph.replay()
    torch.testing.assert_close(indexes, int_tensor([5 * window + 2, window + 6, 4 * window - 1]), atol=0, rtol=0)
