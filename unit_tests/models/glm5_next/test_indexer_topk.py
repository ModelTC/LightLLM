import pytest
import torch

from lightllm.models.glm5_next.indexer import HAS_VLLM, Glm5NextNsaInfer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def indexer():
    return Glm5NextNsaInfer(
        0, {"index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128, "rms_norm_eps": 1e-5}, 1
    )


def _require_vllm_topk():
    if not HAS_VLLM:
        pytest.skip("vLLM top_k_per_row_decode required")


def _assert_topk(logits, lengths, indices):
    topk = indices.shape[1]
    valid = torch.arange(topk, device=logits.device)[None, :] < lengths[:, None]
    assert (indices[~valid] == -1).all()
    assert (indices[valid] >= 0).all()
    assert (indices < lengths[:, None]).all()
    for row, length in zip(indices, lengths.tolist()):
        count = min(length, topk)
        assert row[:count].unique().numel() == count

    got = logits.gather(1, indices.long().clamp_min(0)).masked_fill(~valid, -float("inf"))
    positions = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill(positions[None, :] >= lengths[:, None], -float("inf"))
    # Compare values, since different equal-valued candidates are valid top-k.
    torch.testing.assert_close(got.sort(descending=True).values, masked.topk(topk).values, rtol=0, atol=0)


@pytest.mark.parametrize("pools", [640, 8192, 65536, 262144])
@pytest.mark.parametrize("distribution", ["normal", "concentrated", "ties"])
def test_topk_variable_lengths(indexer, pools, distribution):
    _require_vllm_topk()
    torch.manual_seed(42)
    # DeepGEMM logits can have a padded row stride, and output is a query chunk.
    logits = torch.randn(8, pools + 128, device="cuda")[:, :pools]
    if distribution == "concentrated":
        logits.mul_(0.001).add_(1)
    elif distribution == "ties":
        logits.copy_(torch.randint(0, 4, logits.shape, device="cuda"))
    lengths = torch.tensor([0, 1, 511, 512, 513, pools // 3, pools - 3, pools], device="cuda", dtype=torch.int32)
    positions = torch.arange(pools, device="cuda")
    logits.masked_fill_(positions[None, :] >= lengths[:, None], float("nan"))
    output = torch.full((10, 512), -2, device="cuda", dtype=torch.int32)
    indices = output[1:-1]
    indexer.select_topk_indices(logits, lengths, indices)
    _assert_topk(logits, lengths, indices)
    assert (output[[0, -1]] == -2).all()


@pytest.mark.parametrize("pools", [8192, 262144])
def test_topk_cuda_graph_with_changing_lengths(indexer, pools):
    _require_vllm_topk()
    logits = torch.zeros(8, pools, device="cuda")
    lengths = torch.zeros(8, device="cuda", dtype=torch.int32)
    indices = torch.empty(8, 512, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        indexer.select_topk_indices(logits, lengths, indices)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexer.select_topk_indices(logits, lengths, indices)

    for shift in range(3):
        logits.normal_().mul_(0.001).add_(1)
        lengths.copy_(
            torch.tensor([0, 1, 511, 512, 513, pools // 3, pools - 3, pools], device="cuda", dtype=torch.int32).roll(
                shift
            )
        )
        logits.masked_fill_(torch.arange(pools, device="cuda")[None, :] >= lengths[:, None], float("nan"))
        graph.replay()
        _assert_topk(logits, lengths, indices)


def test_topk_without_vllm(indexer, monkeypatch):
    from lightllm.models.glm5_next import indexer as indexer_module

    monkeypatch.setattr(indexer_module, "HAS_VLLM", False)
    logits = torch.randn(5, 640, device="cuda")
    lengths = torch.tensor([0, 1, 511, 512, 639], device="cuda", dtype=torch.int32)
    indices = torch.empty(5, 512, device="cuda", dtype=torch.int32)
    indexer.select_topk_indices(logits, lengths, indices)
    _assert_topk(logits, lengths, indices)
