from types import SimpleNamespace

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


@pytest.mark.parametrize("mtp_size", [1, 3, 6])
@pytest.mark.parametrize("max_pools", [1024, 262144])
def test_paged_decode_matches_nonpaged_with_graph_and_mtp(indexer, monkeypatch, mtp_size, max_pools):
    pytest.importorskip("deep_gemm")
    _require_vllm_topk()
    torch.manual_seed(27)
    rows = 5 * mtp_size
    storage = torch.empty(1024, 1, 584, device="cuda", dtype=torch.bfloat16)
    packed = storage.view(torch.uint8)[:, :, -132:]
    packed[:, 0, :128] = torch.randn(1024, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.pow(2.0, torch.randint(-3, 2, (1024,), device="cuda").float())
    packed[:, 0, 128:] = scales.view(torch.uint8).view(-1, 4)
    table = torch.full((5, max_pools * 4), -1, device="cuda", dtype=torch.int32)
    table[:, 3::4] = torch.randint(0, 1024, (5, max_pools), device="cuda", dtype=torch.int32)
    req_idx = torch.tensor([1, 3, 0, 2, 4], device="cuda", dtype=torch.int32).repeat_interleave(mtp_size)
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, device="cuda") * 0.05
    infer_state = SimpleNamespace(
        b_req_idx=req_idx,
        b_seq_len=lengths,
        b1_cu_q_seq_len=torch.arange(rows + 1, device="cuda", dtype=torch.int32),
        max_q_seq_len=1,
        req_manager=SimpleNamespace(req_to_token_indexs=table),
    )
    att_state = SimpleNamespace(lengths=lengths)
    logits_outputs = []
    select_topk = indexer.select_topk_indices

    def capture_logits(logits, pool_lengths, indices):
        logits_outputs.append(logits)
        select_topk(logits, pool_lengths, indices)

    monkeypatch.setattr(indexer, "select_topk_indices", capture_logits)

    def run():
        return indexer._get_decode_indices(q, weights, packed, infer_state, att_state, max_pools)

    # Match startup capture with only empty HOLD rows, then replay real work.
    run()
    logits_outputs.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    graph_logits = logits_outputs.pop()
    for bases in ([0, 2047, 2051, max_pools * 4 - mtp_size + 1, 0], [7, 65, 511, 2053, 0]):
        seqs = torch.tensor(bases, device="cuda", dtype=torch.int32)[:, None]
        seqs = seqs + torch.arange(mtp_size, device="cuda", dtype=torch.int32)[None, :]
        seqs[-1].zero_()
        lengths.copy_(seqs.flatten())
        graph.replay()
        indexer._get_prefill_indices(q, weights, packed, infer_state, att_state, max_pools)
        reference_logits = logits_outputs.pop()
        pool_lengths = lengths // 4
        valid = torch.arange(max_pools, device="cuda")[None, :] < pool_lengths[:, None]
        torch.testing.assert_close(graph_logits[valid], reference_logits[valid], rtol=0, atol=0)
        # Fragmented tables deliberately repeat keys; equal-score top-k ties
        # may choose different indices. Check exact selected values and uniqueness.
        _assert_topk(reference_logits, pool_lengths, actual)
        assert (actual[-mtp_size:] == -1).all()
        # Exercise request-slot reuse and speculative rollback on the same graph.
        req_idx.copy_(req_idx.roll(mtp_size))
