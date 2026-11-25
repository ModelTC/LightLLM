import pytest
import torch

from lightllm.models.deepseek_v41.triton_kernel.attention import pool_pairs, select_indexer_topk


@pytest.mark.parametrize("split", [1, 3, 5])
def test_c2_carries_partial_groups_across_chunk_boundaries(split):
    kv, scores = torch.randn(7, 1024, device="cuda").split(512, -1)
    positions = torch.arange(7, dtype=torch.int32, device="cuda")
    req = torch.zeros(7, dtype=torch.int32, device="cuda")
    state = torch.zeros(2, 2, 2, 512, device="cuda")
    first = pool_pairs(kv[:split], scores[:split], req[:split], positions[:split], state, 1)
    second = pool_pairs(kv[split:], scores[split:], req[split:], positions[split:], state, 1)
    output = torch.cat((first, second))
    assert output[::2].count_nonzero() == 0
    actual = output[1::2]
    expected = (kv[:6].view(3, 2, 512) * scores[:6].view(3, 2, 512).softmax(1)).sum(1).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.008)
    torch.testing.assert_close(state[0, 0, 0], kv[-1])
    torch.testing.assert_close(state[0, 0, 1], scores[-1])


def test_c2_does_not_cross_request_boundaries_or_commit_hold():
    kv = torch.randn(4, 512, device="cuda")
    scores = torch.randn_like(kv)
    state = torch.randn(3, 2, 2, 512, device="cuda")
    previous = state.clone()
    req = torch.tensor([0, 0, 1, 2], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda")
    out = pool_pairs(kv, scores, req, pos, state, 2)
    expected = previous[1, 0, 0] + (kv[2] - previous[1, 0, 0]) * torch.sigmoid(scores[2] - previous[1, 0, 1])
    torch.testing.assert_close(out[2], expected.bfloat16(), atol=0.015625, rtol=0.008)
    torch.testing.assert_close(state[2], previous[2])


def test_shifted_mhc_uses_previous_pre_and_returns_new_coefficients(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.mhc import pre_norm

    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", True)
    t, h = 3, 128
    x = torch.randn(t, 4 * h, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(24, 4 * h, device="cuda") * 0.01
    scale, base = torch.randn(3, device="cuda"), torch.randn(24, device="cuda")
    norm = torch.randn(h, dtype=torch.bfloat16, device="cuda")
    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").expand(t, -1).contiguous()
    actual, _, _, new_pre = pre_norm.hc_pre_norm(
        x, fn, scale, base, norm, 4, 1e-20, 1e-20, 1e-6, 20, previous_pre=previous
    )
    selected = x.view(t, 4, h)[:, 0].float()
    expected = (selected * torch.rsqrt(selected.square().mean(-1, keepdim=True) + 1e-20) * norm).bfloat16()
    projection = x.float() @ fn.T * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
    expected_pre = torch.sigmoid(projection[:, :4] * scale[0] + base[:4]) + 1e-6
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
    torch.testing.assert_close(new_pre, expected_pre, atol=1e-6, rtol=1e-5)


def test_engram_host_embedding_uses_fp8_scales_and_zeroes_other_tp_rows(monkeypatch):
    from lightllm.models.deepseek_v41.engram import EngramEmbeddingWeight

    monkeypatch.setenv("LIGHTLLM_DP_SIZE", "1")
    for key in ("GLOBAL_WORLD_SIZE", "DP_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + key, "2")
    embedding = EngramEmbeddingWeight("layers.1.engram.embed", rows=8, dim=64)
    raw = torch.randn(8, 64).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-4, 0, (8, 2)).float()).to(torch.float8_e8m0fnu)
    embedding.load_hf_weights({embedding.prefix + ".scale": scale})
    assert not embedding.verify_load()
    embedding.load_hf_weights({embedding.prefix + ".weight": raw})
    assert embedding.verify_load()
    assert embedding.verify_load()  # Rechecking a loaded weight keeps its existing CUDA mapping.
    # Inspect this rank's contribution before the normal TP sum.
    embedding.tp_world_size_ = 1
    ids = torch.tensor([[0, 3, 4, 7]], dtype=torch.int64, device="cuda")
    actual = embedding.lookup(ids, group=None).view(4, 64)
    expected = (raw.float() * scale.float().repeat_interleave(32, -1)).bfloat16().cuda()
    torch.testing.assert_close(actual[:2], expected[[0, 3]], atol=0, rtol=0)
    assert actual[2:].count_nonzero() == 0


def test_reindex_restricts_to_candidates_and_always_keeps_latest_block():
    logits = torch.arange(1, 33, dtype=torch.float32).reshape(1, -1)
    # The middle block has the highest valid score. The last block must still be offered to reindex.
    logits[:, 8:16] += 100
    pages = torch.tensor([[7, 5, 9, 3]], dtype=torch.int32)
    valid = torch.tensor([26], dtype=torch.int32)
    mask = torch.empty((1, 4), dtype=torch.bool)
    slots = torch.empty((1, 5), dtype=torch.int32)
    lengths = torch.empty(1, dtype=torch.int32)
    select_indexer_topk(logits, valid, pages, slots, lengths, 8, mask, 2)
    assert mask.tolist() == [[False, True, False, True]]
    assert slots.tolist() == [[43, 44, 45, 46, 47]]
    deeper_scores = torch.arange(32, 0, -1, dtype=torch.float32).view(1, -1)
    select_indexer_topk(deeper_scores, valid, pages, slots, lengths, 8, mask, None)
    assert slots.tolist() == [[40, 41, 42, 43, 44]]
    assert lengths.item() == 5
    select_indexer_topk(logits, torch.zeros_like(valid), pages, slots, lengths, 8, None, None)
    assert slots.tolist() == [[-1] * 5]
    assert lengths.item() == 0


@pytest.mark.parametrize("round_scale", [False, True])
def test_indexer_query_and_cached_key_use_the_same_fp8_scale(round_scale):
    from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k import destindex_copy_indexer_k
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_indexer_rope_hadamard_quant

    vectors = torch.linspace(-1.5, 1.5, 128, device="cuda", dtype=torch.bfloat16).view(1, 128)
    pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    freqs = torch.ones((1, 32), dtype=torch.complex64, device="cuda")
    weights = torch.ones((1, 1), device="cuda")
    query, query_scale = fused_q_indexer_rope_hadamard_quant(
        vectors.unsqueeze(1), weights, 1.0, freqs, pos, apply_hadamard=False, round_scale_to_power_of_two=round_scale
    )
    cache = torch.empty((1, 64 * (128 + 4)), dtype=torch.uint8, device="cuda")
    destindex_copy_indexer_k(
        vectors, pos, pos, cache, 64, compress_ratio=1, freqs_cis=freqs, round_scale_to_power_of_two=round_scale
    )
    expected_scale = 1 / 256 if round_scale else 1.5 / 448
    assert query_scale.item() == pytest.approx(expected_scale)
    assert cache[0, 64 * 128 :].view(torch.float32)[0].item() == pytest.approx(expected_scale)
    expected = (vectors.float() / expected_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(query.view(1, 128).view(torch.uint8), expected)
    assert torch.equal(cache[:, :128], expected)
