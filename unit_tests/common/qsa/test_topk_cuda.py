from dataclasses import replace

import pytest
import torch

from lightllm.common.qsa.indexer import QSAIndexer
from lightllm.common.qsa.topk import cuda_topk_supported

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def components(rows):
    if not cuda_topk_supported(torch.device("cuda", 0), 1027, 512):
        pytest.skip("The optional selector requires its qualified H100 runtime")
    weights = torch.randn((640, 16), device="cuda", dtype=torch.bfloat16)
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    original = QSAIndexer(weights, norm, norm)
    optimized = replace(original, use_cuda_topk=True)
    query = torch.randn((rows, 4, 128), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((16384, 128), device="cuda", dtype=torch.bfloat16)
    requests = torch.arange(rows, device="cuda", dtype=torch.int32) % 9
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    table = torch.randint(0, 16384, (9, 4110), device="cuda", dtype=torch.int32)
    return original, optimized, (query, cache, requests, lengths, table)


@pytest.mark.parametrize("rows", [3, 24])
@pytest.mark.parametrize("visible", [0, 3, 2047, 2048, 2051, 4096])
def test_optional_selector_preserves_unsorted_slot_order(rows, visible):
    original, optimized, data = components(rows)
    data[3].fill_(visible)
    assert torch.equal(original.select(*data), optimized.select(*data))


@pytest.mark.parametrize("rows", [3, 24])
def test_short_long_graph_transitions_ignore_poisoned_short_scores(rows, monkeypatch):
    original, optimized, data = components(rows)
    allocate = torch.empty

    def poisoned_empty(*args, **kwargs):
        value = allocate(*args, **kwargs)
        if value.is_cuda and value.dtype == torch.float32 and value.ndim == 2 and value.shape[1] == 1027:
            value.fill_(float("nan"))
        return value

    monkeypatch.setattr(torch, "empty", poisoned_empty)
    data[3].fill_(4096)
    original.select(*data)
    optimized.select(*data)
    torch.cuda.synchronize()
    before, after = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(before):
        expected = original.select(*data)
    with torch.cuda.graph(after):
        actual = optimized.select(*data)
    limits = torch.tensor([0, 3, 2047, 2048, 2051, 4096], device="cuda", dtype=torch.int32)
    for step in range(24):
        data[0].normal_()
        data[3].copy_(limits[(torch.arange(rows, device="cuda") + step) % limits.numel()])
        before.replay()
        after.replay()
        assert torch.equal(actual, expected)
