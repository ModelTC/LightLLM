from dataclasses import replace

import pytest
import torch

from lightllm.common.qsa.indexer import QSAIndexer
from lightllm.common.qsa.topk import flashinfer_topk_indices, flashinfer_topk_supported

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def require_ragged_topk():
    if not flashinfer_topk_supported(torch.device("cuda", 0), 65542, 512):
        pytest.skip("FlashInfer graph-safe TopK on Hopper is required")


def test_ragged_graph_uses_visible_lengths_and_chunk_offset():
    rows, columns, k, first = 17, 65542, 512, 5
    scores = torch.randn(rows, columns, device="cuda")
    visible = torch.zeros(first + rows + 3, dtype=torch.int32, device="cuda")
    lengths = visible[first : first + rows]
    limits = torch.tensor([0, 3, 2047, 2048, 2051, 65536, 262144], dtype=torch.int32, device="cuda")
    cols = torch.arange(columns, device="cuda")

    def select():
        return flashinfer_topk_indices(scores, k, visible_lengths=visible, first=first)

    select()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = select()
    for step in range(12):
        lengths.copy_(limits[(torch.arange(rows, device="cuda") + step) % limits.numel()])
        scores.normal_()
        if step % 3 == 0:
            scores.round_()
        scores.masked_fill_(cols[None, :] >= lengths[:, None] // 4, 1e9)
        scores.masked_fill_((lengths <= k * 4)[:, None], float("nan"))
        graph.replay()
        for row, complete in enumerate((lengths // 4).tolist()):
            actual = selected[row]
            if complete <= k:
                expected = torch.arange(k, dtype=torch.int32, device="cuda")
                expected[expected >= complete] = -1
                assert torch.equal(actual, expected)
            else:
                assert bool(((actual >= 0) & (actual < complete)).all())
                assert torch.unique(actual).numel() == k
                expected = torch.topk(scores[row, :complete], k).values
                values = scores[row, actual.long()].sort(descending=True).values
                torch.testing.assert_close(values, expected, atol=0, rtol=0)
                ranked = torch.argsort(scores[row, :complete], descending=True, stable=True)[:k]
                assert torch.equal(actual.sort().values.long(), ranked.sort().values)
        expected_order = selected.clone()
        graph.replay()
        torch.testing.assert_close(selected, expected_order, atol=0, rtol=0)


@pytest.mark.parametrize("chunk_size", [3, 64])
def test_indexer_ragged_topk_preserves_slots_across_graph_transitions(chunk_size):
    rows, capacity = 17, 262169
    norm = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    original = QSAIndexer(torch.randn(640, 16, dtype=torch.bfloat16, device="cuda"), norm, norm)
    optimized = replace(original, use_cuda_topk=True)
    cache = torch.zeros(capacity + 3, 128, dtype=torch.bfloat16, device="cuda")
    bits = torch.arange(17, device="cuda")
    cache[:, :17] = ((torch.arange(cache.shape[0], device="cuda")[:, None] // 4 >> bits) & 1).to(cache.dtype)
    query = torch.zeros(rows, 4, 128, dtype=cache.dtype, device="cuda")
    query[:, :, :17] = (2.0 ** bits).to(cache.dtype)
    table = torch.arange(capacity, dtype=torch.int32, device="cuda").expand(2, -1).contiguous()
    requests = torch.arange(rows, dtype=torch.int32, device="cuda") // 4 % 2
    lengths = torch.full((rows,), 262144, dtype=torch.int32, device="cuda")
    data = query, cache, requests, lengths, table
    original.select(*data, query_chunk_size=chunk_size)
    optimized.select(*data, query_chunk_size=chunk_size)
    before, after = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(before):
        expected = original.select(*data, query_chunk_size=chunk_size)
    with torch.cuda.graph(after):
        actual = optimized.select(*data, query_chunk_size=chunk_size)
    limits = torch.tensor([0, 3, 2047, 2048, 2051, 65536, 262144], dtype=torch.int32, device="cuda")
    for step in range(12):
        lengths.copy_(limits[(torch.arange(rows, device="cuda") + step) % limits.numel()])
        requests.copy_((torch.arange(rows, device="cuda") + step) % 2)
        before.replay()
        after.replay()
        assert torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values)
