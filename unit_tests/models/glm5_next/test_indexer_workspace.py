import weakref
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer, get_prefill_logits_chunk_size


@pytest.mark.parametrize("budget_mib,expected", [(64, 64), (128, 128), (256, 256), (512, 512)])
def test_one_million_context_query_parallelism(budget_mib, expected):
    assert get_prefill_logits_chunk_size(2048, 262144, budget_mib) == expected


@pytest.mark.parametrize("num_q_tokens,max_pools", [(1, 262144), (8, 8192), (2048, 640), (8192, 262144), (0, 262144)])
def test_default_matches_previous_bound(num_q_tokens, max_pools):
    assert get_prefill_logits_chunk_size(num_q_tokens, max_pools, 64) == max(
        1, min(num_q_tokens, 16 * 1024 * 1024 // max_pools)
    )


@pytest.mark.parametrize("budget_mib", [0, -1, 15, 1025])
def test_reject_unbounded_workspace(budget_mib):
    with pytest.raises(ValueError, match="budget"):
        get_prefill_logits_chunk_size(2048, 262144, budget_mib)


def test_environment_controls_each_indexer(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_GLM53_PREFILL_LOGITS_MIB", "256")
    indexer = Glm5NextNsaInfer(
        0, {"index_kpool": 4, "index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128, "rms_norm_eps": 1e-5}, 1
    )
    assert indexer.prefill_logits_budget_mib == 256


@pytest.mark.parametrize("max_pools", [0, -1])
def test_reject_invalid_pool_capacity(max_pools):
    with pytest.raises(ValueError, match="max_pools"):
        get_prefill_logits_chunk_size(128, max_pools, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the GLM GPU indexer kernels")
def test_workspace_preserves_mixed_request_pool_selection(monkeypatch):
    """Exercise an unaligned request boundary and a full 1M key range."""
    if torch.cuda.mem_get_info()[0] < 1536 * 1024 ** 2:
        pytest.skip("requires at least 1.5 GiB of unused GPU memory")
    import deep_gemm

    fp8_mqa_logits = deep_gemm.fp8_mqa_logits
    previous_logits = None

    def track_logits_lifetime(*args, **kwargs):
        nonlocal previous_logits
        if previous_logits is not None:
            assert previous_logits() is None, "the previous logits chunk is still alive"
        logits = fp8_mqa_logits(*args, **kwargs)
        previous_logits = weakref.ref(logits)
        return logits

    monkeypatch.setattr(deep_gemm, "fp8_mqa_logits", track_logits_lifetime)
    monkeypatch.setenv("LIGHTLLM_GLM53_PREFILL_LOGITS_MIB", "64")
    torch.manual_seed(20261029)
    max_pools = 262144
    packed_buffer = torch.empty((max_pools * 2, 1, 132), dtype=torch.uint8, device="cuda")
    packed_buffer[:, 0, :128] = torch.randn(max_pools * 2, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    packed_buffer[:, 0, 128:] = torch.ones(max_pools * 2, device="cuda").view(torch.uint8).view(-1, 4)
    req_to_token_indexs = torch.full((2, 1048576), -1, dtype=torch.int32, device="cuda")
    for batch in range(2):
        req_to_token_indexs[batch, 3::4] = torch.arange(
            batch * max_pools, (batch + 1) * max_pools, dtype=torch.int32, device="cuda"
        )
    q_fp8 = torch.randn(128, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(128, 32, device="cuda") * 0.05
    lengths = torch.cat(
        [torch.arange(65536 - 30, 65537, device="cuda"), torch.arange(1048576 - 96, 1048577, device="cuda")]
    ).to(torch.int32)
    infer_state = SimpleNamespace(
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([65536, 1048576], dtype=torch.int32, device="cuda"),
        b1_cu_q_seq_len=torch.tensor([0, 31, 128], dtype=torch.int32, device="cuda"),
        max_q_seq_len=97,
        req_manager=SimpleNamespace(req_to_token_indexs=req_to_token_indexs),
    )
    att_state = SimpleNamespace(lengths=lengths)
    indexer = Glm5NextNsaInfer(
        0, {"index_kpool": 4, "index_topk": 2048, "index_n_heads": 32, "index_head_dim": 128, "rms_norm_eps": 1e-5}, 1
    )
    expected = indexer._get_prefill_indices(q_fp8, weights, packed_buffer, infer_state, att_state, max_pools)
    expected = expected.sort(dim=1).values
    for budget in [256, 512]:
        indexer.prefill_logits_budget_mib = budget
        actual = indexer._get_prefill_indices(q_fp8, weights, packed_buffer, infer_state, att_state, max_pools)
        torch.testing.assert_close(actual.sort(dim=1).values, expected, rtol=0, atol=0)
        assert ((actual >= 0) & (actual < (lengths // 4)[:, None])).all().item()
