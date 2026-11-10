import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the GLM GPU indexer kernels")
@pytest.mark.parametrize(
    "max_pools,q_lengths,kv_lengths,expected_chunks",
    [(640, [2047, 2048, 2049] * 4, [2304, 2432, 2560] * 4, 2), (262144, [31, 1026], [65536, 1048576], 17)],
)
def test_workspace_preserves_mixed_request_pool_selection(
    monkeypatch, max_pools, q_lengths, kv_lengths, expected_chunks
):
    """Chunked prefill bounds padded allocations and preserves top-k results."""
    import deep_gemm

    indexer = Glm5NextNsaInfer(
        0, {"index_kpool": 4, "index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128, "rms_norm_eps": 1e-5}, 1
    )
    fp8_mqa_logits = deep_gemm.fp8_mqa_logits
    previous_logits = None
    chunk_count = 0

    def track_logits_lifetime(*args, **kwargs):
        nonlocal previous_logits, chunk_count
        if previous_logits is not None:
            assert previous_logits() is None, "the previous logits chunk is still alive"
        logits = fp8_mqa_logits(*args, **kwargs)
        allocated_bytes = logits.untyped_storage().nbytes()
        assert allocated_bytes <= 64 * 1024 ** 2
        previous_logits = weakref.ref(logits)
        chunk_count += 1
        return logits

    monkeypatch.setattr(deep_gemm, "fp8_mqa_logits", track_logits_lifetime)
    torch.manual_seed(20261029)
    batch, tokens = len(q_lengths), sum(q_lengths)
    packed_buffer = torch.empty((batch * max_pools, 1, 132), dtype=torch.uint8, device="cuda")
    packed_buffer[:, 0, :128] = (
        torch.randn(batch * max_pools, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    )
    packed_buffer[:, 0, 128:] = torch.ones(batch * max_pools, device="cuda").view(torch.uint8).view(-1, 4)
    req_to_token_indexs = torch.full((batch, max_pools * 4), -1, dtype=torch.int32, device="cuda")
    req_to_token_indexs[:, 3::4] = torch.arange(batch * max_pools, dtype=torch.int32, device="cuda").view(batch, -1)
    q_fp8 = torch.randn(tokens, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(tokens, 32, device="cuda") * 0.05
    lengths = torch.cat([torch.arange(k - q + 1, k + 1, device="cuda") for q, k in zip(q_lengths, kv_lengths)]).int()
    infer_state = SimpleNamespace(
        b_req_idx=torch.arange(batch, dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor(kv_lengths, dtype=torch.int32, device="cuda"),
        b1_cu_q_seq_len=torch.tensor([0] + q_lengths, device="cuda").cumsum(0, dtype=torch.int32),
        max_q_seq_len=max(q_lengths),
        req_manager=SimpleNamespace(req_to_token_indexs=req_to_token_indexs),
    )
    att_state = SimpleNamespace(lengths=lengths)
    expected = indexer._get_prefill_indices(q_fp8, weights, packed_buffer, infer_state, att_state, max_pools)
    assert chunk_count == expected_chunks
    expected = expected.sort(dim=1).values
    chunk_size = indexer._get_prefill_chunk_size(max_pools)
    monkeypatch.setattr(indexer, "_get_prefill_chunk_size", lambda _: chunk_size // 2)
    actual = indexer._get_prefill_indices(q_fp8, weights, packed_buffer, infer_state, att_state, max_pools)
    torch.testing.assert_close(actual.sort(dim=1).values, expected, rtol=0, atol=0)
    valid = actual >= 0
    assert (valid.sum(dim=1) == (lengths // 4).clamp_max(indexer.topk // 4)).all().item()
    assert ((actual == -1) | (valid & (actual < (lengths // 4)[:, None]))).all().item()
