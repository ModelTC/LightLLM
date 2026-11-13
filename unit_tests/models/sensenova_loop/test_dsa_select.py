import pytest
import torch

from lightllm.models.sensenova_loop.triton_kernel.dsa_select import exact_float_topk


def oracle(scores, topk):
    ids = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :topk]
    valid = torch.isfinite(scores.gather(1, ids))
    ids = ids.masked_fill(~valid, scores.shape[1]).sort(dim=1).values
    return ids.masked_fill(ids == scores.shape[1], -1).int()


@pytest.mark.parametrize("distribution", ["signed", "ties", "near_cutoff", "zero", "extremes"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_exact_selector_preserves_lowest_index_ties_and_padding(distribution, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    scores = torch.randn(8, 4112, device=device)[:, :4099]
    if distribution == "ties":
        scores.copy_(torch.randint(-2, 3, scores.shape, device=device))
    elif distribution == "zero":
        scores.zero_()
        scores[:, ::2] = -0.0
    elif distribution == "near_cutoff":
        scores.fill_(1)
        scores[:, 1::3] = torch.nextafter(scores[:, 1::3], torch.full_like(scores[:, 1::3], torch.inf))
    elif distribution == "extremes":
        values = torch.tensor(
            [
                -torch.finfo(torch.float32).max,
                -(2.0 ** -126),
                -(2.0 ** -149),
                -0.0,
                0.0,
                2.0 ** -149,
                2.0 ** -126,
                torch.finfo(torch.float32).max,
            ],
            device=device,
        )
        scores.copy_(values[torch.arange(scores.shape[1], device=device) % values.numel()])
    lengths = torch.tensor([0, 1, 2047, 2048, 2049, 3001, 4098, 4099], device=device, dtype=torch.int32)
    scores.masked_fill_(torch.arange(scores.shape[1], device=device)[None, :] >= lengths[:, None], -torch.inf)
    expected = oracle(scores, 2048)
    for backend in ["torch", "radix", "auto"] if device == "cuda" else ["torch"]:
        actual = exact_float_topk(scores, lengths, 2048, backend=backend)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selector_replay_changes_lengths_and_scores_on_nondefault_stream():
    scores = torch.empty(4, 8192, device="cuda")
    lengths = torch.zeros(4, device="cuda", dtype=torch.int32)
    scores.fill_(-torch.inf)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        exact_float_topk(scores, lengths, 2048)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = exact_float_topk(scores, lengths, 2048)
    for shift in range(3):
        lengths.copy_(torch.tensor([0, 1, 2047, 8192], device="cuda", dtype=torch.int32).roll(shift))
        scores.copy_(torch.randint(-3, 4, scores.shape, device="cuda"))
        scores.masked_fill_(torch.arange(8192, device="cuda")[None, :] >= lengths[:, None], -torch.inf)
        graph.replay()
        torch.testing.assert_close(output, oracle(scores, 2048), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flashinfer_page_topk_exact_mapping_and_short_lengths():
    try:
        from lightllm.models.sensenova_loop.triton_kernel.dsa_perf import (
            flashinfer_topk_page_table,
            flashinfer_topk_page_table_available,
        )
    except (ImportError, OSError, RuntimeError):
        pytest.skip("FlashInfer is unavailable")
    if not flashinfer_topk_page_table_available():
        pytest.skip("FlashInfer fused page TopK is unavailable")

    torch.manual_seed(123)
    batch, keys, topk, requests_count = 8, 4096, 2048, 3
    scores = torch.randn(batch, keys, device="cuda", dtype=torch.float32)
    scores[:, ::113] = 2.0
    scores[:, ::127] = 2.0
    requests = torch.tensor([0, 1, 2, 0, 2, 1, 0, 2], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 1, 17, 2047, 2048, 2049, 4095, 4096], device="cuda", dtype=torch.int32)
    table = torch.arange(requests_count * keys, device="cuda", dtype=torch.int32).reshape(requests_count, keys)
    output = torch.empty((batch, topk), device="cuda", dtype=torch.int32)
    flashinfer_topk_page_table(scores, table, requests, lengths, topk, out=output)
    torch.cuda.synchronize()

    for row in range(batch):
        length = int(lengths[row])
        local = torch.argsort(scores[row, :length], descending=True, stable=True)[:topk]
        expected = table[requests[row], local].sort().values
        actual = output[row][output[row] >= 0].sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_radix_topk_fused_page_mapping_exact():
    from lightllm.models.sensenova_loop.triton_kernel.dsa_radix import exact_radix_topk

    torch.manual_seed(321)
    batch, requests_count, keys, topk = 8, 3, 4099, 2048
    scores = torch.randn(batch, keys, device="cuda", dtype=torch.float32)
    scores[:, ::113] = 2.0
    scores[:, ::127] = 2.0
    requests = torch.tensor([0, 1, 2, 0, 2, 1, 0, 2], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 1, 17, 2047, 2048, 2049, 4098, 4099], device="cuda", dtype=torch.int32)
    raw = torch.randperm(requests_count * keys, device="cuda", dtype=torch.int32).reshape(requests_count, keys)
    output = exact_radix_topk(scores, lengths, topk, table=raw, requests=requests)
    torch.cuda.synchronize()

    for row in range(batch):
        length = int(lengths[row])
        local = torch.argsort(scores[row, :length], descending=True, stable=True)[:topk]
        expected = raw[requests[row], local].sort().values
        actual = output[row][output[row] >= 0].sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
