from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.deepseek_v41.triton_kernel.attention import (
    compress_and_normalize_kv_pairs,
    select_indexer_topk_slots,
)
from lightllm.models.deepseek_v41.triton_kernel.q_rope import apply_attention_query_rope


@pytest.mark.parametrize("tokens", [1, 129])
@pytest.mark.parametrize("inplace", [False, True])
def test_q_rope_preserves_nope_and_padded_heads(tokens, inplace):
    heads = 3
    source = torch.randn((tokens, heads, 512), device="cuda", dtype=torch.bfloat16)
    output = torch.full((tokens, 8, 512), 9, device="cuda", dtype=torch.bfloat16)
    q = output[:, :heads]
    if inplace:
        q.copy_(source)
    positions = torch.randint(0, 32, (tokens,), device="cuda", dtype=torch.int32)
    angles = torch.randn((32, 32), device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)

    apply_attention_query_rope(q if inplace else source, q, freqs, positions)

    pairs = torch.view_as_complex(source[..., -64:].float().unflatten(-1, (32, 2)))
    expected = torch.view_as_real(pairs * freqs[positions.long(), None]).flatten(-2).bfloat16()
    torch.testing.assert_close(q[..., -64:], expected, rtol=0.008, atol=1e-6)
    torch.testing.assert_close(q[..., :-64], source[..., :-64], rtol=0, atol=0)
    assert (output[:, heads:] == 9).all()


@pytest.mark.parametrize("split", [1, 3, 5])
def test_c2_carries_partial_groups_across_chunk_boundaries(split):
    kv, scores = torch.randn(7, 1024, device="cuda").split(512, -1)
    positions = torch.arange(7, dtype=torch.int32, device="cuda")
    req = torch.zeros(7, dtype=torch.int32, device="cuda")
    state = torch.zeros(2, 4, 2, 512, device="cuda")
    weight = torch.rand(512, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6
    first = compress_and_normalize_kv_pairs(
        kv[:split], scores[:split], req[:split], positions[:split], state, 1, weight, eps
    )
    second = compress_and_normalize_kv_pairs(
        kv[split:], scores[split:], req[split:], positions[split:], state, 1, weight, eps
    )
    output = torch.cat((first, second))
    assert output[::2].count_nonzero() == 0
    actual = output[1::2]
    expected = (kv[:6].view(3, 2, 512) * scores[:6].view(3, 2, 512).softmax(1)).sum(1).bfloat16()
    torch.testing.assert_close(actual, rmsnorm_forward(expected, weight, eps), atol=0.015625, rtol=0.008)
    torch.testing.assert_close(state[0, 0, 0], kv[-1])
    torch.testing.assert_close(state[0, 0, 1], scores[-1])


def test_c2_does_not_cross_request_boundaries_or_commit_hold():
    kv = torch.randn(4, 512, device="cuda")
    scores = torch.randn_like(kv)
    state = torch.randn(3, 4, 2, 512, device="cuda")
    previous = state.clone()
    req = torch.tensor([0, 0, 1, 2], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda")
    weight = torch.rand(512, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6
    out = compress_and_normalize_kv_pairs(kv, scores, req, pos, state, 2, weight, eps)
    expected = previous[1, 0, 0] + (kv[2] - previous[1, 0, 0]) * torch.sigmoid(scores[2] - previous[1, 0, 1])
    torch.testing.assert_close(
        out[2], rmsnorm_forward(expected.bfloat16()[None], weight, eps)[0], atol=0.015625, rtol=0.008
    )
    torch.testing.assert_close(state[2], previous[2])


@pytest.mark.parametrize("disable_deepgemm", [False, True], ids=["deepgemm", "torch"])
def test_shifted_mhc_uses_previous_pre_and_returns_new_coefficients(monkeypatch, disable_deepgemm):
    from lightllm.common.basemodel.triton_kernel.mhc import hc_pre_norm, pre_norm, shifted_hc_pre_norm

    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", disable_deepgemm)
    t, h = 3, 128
    x = torch.randn(t, 4 * h, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(24, 4 * h, device="cuda") * 0.01
    scale, base = torch.randn(3, device="cuda"), torch.randn(24, device="cuda")
    norm = torch.randn(h, dtype=torch.bfloat16, device="cuda")
    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").expand(t, -1).contiguous()
    actual, residual_mix, post_mix, new_pre = shifted_hc_pre_norm(
        x, fn, scale, base, norm, 4, 1e-20, 1e-20, 1e-6, 20, previous_pre=previous
    )
    _, expected_residual_mix, expected_post_mix = hc_pre_norm(x, fn, scale, base, norm, 4, 1e-20, 1e-20, 1e-6, 20)
    selected = x.view(t, 4, h)[:, 0].float()
    expected = (selected * torch.rsqrt(selected.square().mean(-1, keepdim=True) + 1e-20) * norm).bfloat16()
    projection = x.float() @ fn.T * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
    expected_pre = torch.sigmoid(projection[:, :4] * scale[0] + base[:4]) + 1e-6
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
    torch.testing.assert_close(residual_mix, expected_residual_mix)
    torch.testing.assert_close(post_mix, expected_post_mix)
    torch.testing.assert_close(new_pre, expected_pre, atol=1e-4, rtol=1e-5)


def test_engram_host_embedding_uses_fp8_scales_and_zeroes_other_tp_rows(monkeypatch):
    from lightllm.models.deepseek_v41.engram import EngramEmbeddingWeight

    monkeypatch.setenv("LIGHTLLM_DP_SIZE", "1")
    for key in ("GLOBAL_WORLD_SIZE", "DP_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + key, "2")
    embedding = EngramEmbeddingWeight("layers.1.engram.embed", num_embeddings=8, embedding_dim=64)
    raw = torch.randn(8, 64).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-4, 0, (8, 2)).float()).to(torch.float8_e8m0fnu)
    embedding.load_hf_weights({embedding.weight_prefix + ".scale": scale})
    assert not embedding.verify_load()
    embedding.load_hf_weights({embedding.weight_prefix + ".weight": raw})
    assert embedding.verify_load()
    assert embedding.verify_load()  # Rechecking a loaded weight keeps its existing CUDA mapping.
    # Inspect this rank's contribution before the normal TP sum.
    embedding.tp_world_size_ = 1
    ids = torch.tensor([[0, 3, 4, 7]], dtype=torch.int64, device="cuda")
    actual = embedding.lookup_rows(ids, tp_group=None).view(4, 64)
    expected = (raw.float() * scale.float().repeat_interleave(32, -1)).bfloat16().cuda()
    torch.testing.assert_close(actual[:2], expected[[0, 3]], atol=0, rtol=0)
    assert actual[2:].count_nonzero() == 0


def _torch_indexer_topk(logits, lengths, pages, slots, selected_lengths, page_size, candidate_mask, candidate_topk):
    """Keep the latest candidate block, then map finite causal top-k entries to physical slots."""
    columns = torch.arange(logits.shape[1], device=logits.device)
    logits.masked_fill_(columns >= lengths[:, None], -torch.inf)
    if candidate_mask is not None:
        blocks = logits.unflatten(-1, (-1, 8))
        if candidate_topk is not None:
            scores = blocks.amax(-1)
            recent = ((lengths.long() - 1).clamp_min(0) // 8).unsqueeze(1)
            scores.scatter_(1, recent, torch.inf)
            selected = scores.topk(min(candidate_topk, scores.shape[1]), sorted=False).indices
            candidate_mask.zero_().scatter_(1, selected, True)
        else:
            blocks.masked_fill_(~candidate_mask[..., None], -torch.inf)
    scores, indices = logits.topk(min(slots.shape[1], logits.shape[1]), sorted=False)
    indices = indices.masked_fill(~scores.isfinite(), logits.shape[1]).sort(-1).values
    valid = indices < logits.shape[1]
    indices = indices.masked_fill(~valid, 0)
    physical = pages.gather(1, indices // page_size) * page_size + indices % page_size
    slots.fill_(-1)
    slots[:, : indices.shape[1]] = physical.masked_fill(~valid, -1)
    selected_lengths.copy_(valid.sum(-1))


def test_reindex_restricts_to_candidates_and_always_keeps_latest_block():
    logits = torch.arange(1, 33, device="cuda", dtype=torch.float32).reshape(1, -1)
    # The middle block has the highest valid score. The last block must still be offered to reindex.
    logits[:, 8:16] += 100
    pages = torch.tensor([[7, 5, 9, 3]], device="cuda", dtype=torch.int32)
    valid = torch.tensor([26], device="cuda", dtype=torch.int32)
    mask = torch.empty((1, 4), device="cuda", dtype=torch.bool)
    slots = torch.empty((1, 5), device="cuda", dtype=torch.int32)
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    select_indexer_topk_slots(logits, valid, pages, slots, lengths, 8, mask, 2)
    assert mask.tolist() == [[False, True, False, True]]
    assert slots.tolist() == [[43, 44, 45, 46, 47]]
    deeper_scores = torch.arange(32, 0, -1, device="cuda", dtype=torch.float32).view(1, -1)
    select_indexer_topk_slots(deeper_scores, valid, pages, slots, lengths, 8, mask, None)
    assert slots.tolist() == [[40, 41, 42, 43, 44]]
    assert lengths.item() == 5
    select_indexer_topk_slots(logits, torch.zeros_like(valid), pages, slots, lengths, 8, None, None)
    assert slots.tolist() == [[-1] * 5]
    assert lengths.item() == 0


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_indexer_query_and_cached_key_use_the_same_fp8_scale(compress_ratio):
    from lightllm.models.deepseek_v41.triton_kernel.indexer import (
        rotate_and_cache_indexer_keys,
        rotate_and_quantize_indexer_queries,
    )

    vectors = torch.linspace(-1.5, 1.5, 128, device="cuda", dtype=torch.bfloat16).view(1, 128)
    pos = torch.zeros(1, dtype=torch.int32, device="cuda")
    freqs = torch.ones((1, 32), dtype=torch.complex64, device="cuda")
    weights = torch.ones((1, 1), device="cuda")
    query = torch.empty((1, 1, 128), dtype=torch.float8_e4m3fn, device="cuda")
    scaled_head_weights = torch.empty((1, 1, 1), dtype=torch.float32, device="cuda")
    rotate_and_quantize_indexer_queries(
        vectors.unsqueeze(1), weights, 1.0, freqs, pos, query_fp8_out=query, scaled_head_weights_out=scaled_head_weights
    )
    cache = torch.empty((1, 64 * (128 + 4)), dtype=torch.uint8, device="cuda")
    rotate_and_cache_indexer_keys(vectors, freqs, pos, pos + compress_ratio - 1, cache, compress_ratio, 64)
    expected_scale = 1 / 256
    assert scaled_head_weights.item() == pytest.approx(expected_scale)
    assert cache[0, 64 * 128 :].view(torch.float32)[0].item() == pytest.approx(expected_scale)
    expected = (vectors.float() / expected_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(query.view(1, 128).view(torch.uint8), expected)
    assert torch.equal(cache[:, :128], expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [16, 5120])
@pytest.mark.parametrize("masked", [False, True])
def test_engram_gate_matches_reference_and_cuda_graph(dtype, dim, masked):
    from lightllm.models.deepseek_v41.engram import apply_engram_to_streams

    tokens, mult = 7, 4
    # Both producers may expose row slices; no contiguous copies are needed.
    streams = torch.randn(tokens * 2, mult, dim, device="cuda", dtype=dtype)[::2]
    kv = torch.randn(tokens * 2, (mult + 1) * dim, device="cuda", dtype=dtype)[::2]
    streams[0].zero_()
    kv[0, : mult * dim].zero_()
    weight = SimpleNamespace(
        engram_embed_=SimpleNamespace(lookup_rows=lambda _, tp_group: None),
        engram_wkv_=SimpleNamespace(mm=lambda _: kv),
        engram_q_=SimpleNamespace(weight=torch.randn(mult, dim, device="cuda", dtype=dtype)),
        engram_k_=SimpleNamespace(weight=torch.randn(mult, dim, device="cuda", dtype=dtype)),
    )
    token_mask = torch.arange(tokens, device="cuda") % 2 == 0 if masked else None
    eps = 1e-20

    def reference():
        h = streams.float()
        key = kv[:, : mult * dim].float().reshape(tokens, mult, dim)
        value = kv[:, mult * dim :].float()
        dot = (h * key * weight.engram_q_.weight * weight.engram_k_.weight).sum(-1)
        dot *= torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps) * dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = torch.where(token_mask[:, None], gate, 0)
        return (h + gate[..., None] * value[:, None, :]).to(dtype)

    def check(actual):
        torch.testing.assert_close(actual, reference(), atol=0.015625 if dtype == torch.bfloat16 else 2e-6, rtol=1e-5)
        if masked:
            torch.testing.assert_close(actual[~token_mask], streams[~token_mask], atol=0, rtol=0)

    check(apply_engram_to_streams(streams, None, weight, eps, None, token_mask))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = apply_engram_to_streams(streams, None, weight, eps, None, token_mask)
    streams.normal_()
    kv.normal_()
    graph.replay()
    check(actual)


@pytest.mark.parametrize("topk,width", [(5, 32), (128, 64), (512, 2112)])
@pytest.mark.parametrize("mode", ["dense", "source", "all_blocks", "reindex"])
def test_indexer_graph_matches_torch(topk, width, mode):
    rows, blocks = 5, width // 8
    # Recreate the row stride of a sliced DeepGEMM logits buffer.
    logits = torch.empty(rows, width + 32, device="cuda")[:, :width]
    pages = torch.randperm(rows * blocks, device="cuda", dtype=torch.int32).reshape(rows, blocks)
    lengths = torch.tensor([0, 1, 9, width - 3, width], device="cuda", dtype=torch.int32)
    mask = None if mode == "dense" else torch.empty(rows, blocks, dtype=torch.bool, device="cuda")
    candidate_topk = {"source": 2, "all_blocks": blocks + 1}.get(mode)
    slots = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    counts = torch.empty_like(lengths)

    def run():
        select_indexer_topk_slots(logits, lengths, pages, slots, counts, 8, mask, candidate_topk)

    logits.normal_()
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        # Unique scores make the selected set independent of tie breaking.
        logits.copy_(torch.randperm(rows * width, device="cuda").reshape(rows, width))
        logits[:, 2::11] = -torch.inf
        logits[:, 5] = torch.inf
        logits[:, 17] = torch.nan
        lengths.copy_(lengths.roll(1))
        if mask is not None:
            mask.random_(2)
        expected_mask = mask.clone() if mask is not None else None
        expected_slots, expected_counts = torch.empty_like(slots), torch.empty_like(counts)
        _torch_indexer_topk(
            logits.clone(), lengths, pages, expected_slots, expected_counts, 8, expected_mask, candidate_topk
        )
        graph.replay()
        torch.testing.assert_close(slots, expected_slots)
        torch.testing.assert_close(counts, expected_counts)
        if mask is not None:
            torch.testing.assert_close(mask, expected_mask)


@pytest.mark.parametrize("heads,padded", [(8, 64), (64, 64)])
def test_q_rope_in_place_matches_separate_output(heads, padded):
    tokens = 67
    storage = torch.randn(tokens, padded, 512, device="cuda", dtype=torch.bfloat16)
    original = storage.clone()
    active = storage[:, :heads]
    expected = torch.empty_like(active)
    pos = torch.randperm(tokens, device="cuda", dtype=torch.int32)
    freqs = torch.polar(torch.ones(tokens, 32, device="cuda"), torch.randn(tokens, 32, device="cuda"))
    apply_attention_query_rope(original[:, :heads], expected, freqs, pos)
    apply_attention_query_rope(active, active, freqs, pos)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_attention_query_rope(active, active, freqs, pos)
    storage.copy_(original)
    graph.replay()
    torch.testing.assert_close(active, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage[:, heads:], original[:, heads:], rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [0, 1, 6, 17, 513])
@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.parametrize("vision", [False, True])
def test_engram_fused_hash_matches_torch_for_ragged_inputs_and_graph(monkeypatch, tokens, heads, vision):
    from lightllm.models.deepseek_v41 import engram

    config = dict(
        vision_n_layers=int(vision),
        engram_compressed_vocab_size=64,
        engram_pad_token_id=2,
        engram_max_ngram_size=4,
        engram_layer_ids=[1, 14],
        engram_vocab_size=101,
        engram_n_heads=heads,
    )
    monkeypatch.setattr(engram, "_build_compressed_token_id_map", lambda _: (list(range(64)), 64))
    length = tokens + 132
    table = torch.randperm(3 * length, device="cuda").int().view(3, length)
    hasher = engram.EngramHasher(config, None, table.numel())
    history = hasher.compressed_token_ids_by_slot
    history.random_(64)
    history[table[:, 126].long()] = -1 if vision else 0
    row = torch.arange(tokens, device="cuda")
    req = (row % 3).int()
    position = (row // 3 + torch.tensor([0, 127, 130], device="cuda")[req.long()]).int()
    ids = torch.randint(64, (tokens,), device="cuda")
    if vision:
        ids[::5] = 1000
    state = SimpleNamespace(
        input_ids=ids,
        position_ids=position,
        dsv4_sparse_req_idx=req,
        mem_index=table[req.long(), position.long()],
        req_manager=SimpleNamespace(req_to_token_indexs=table),
    )
    before = history.clone()

    def reference():
        history[state.mem_index.long()] = torch.where(ids < 64, ids, -1)
        positions, reqs = position.long(), req.long()
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        previous_tokens = []
        for shift in range(hasher.max_ngram_size):
            slot = table[reqs, (positions - shift).clamp_min(0)].long()
            source = history[slot]
            blocked |= (positions < shift) | (source == -1)
            previous_tokens.append(torch.where(blocked, hasher.compressed_pad_token_id, source))
        products = torch.stack(previous_tokens, -1)[:, None, :] * hasher.hash_multipliers[None, :, :]
        rolling, hashes = products[..., 0], []
        for shift in range(1, hasher.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling[..., None] % hasher.head_bucket_counts[:, (shift - 1) * heads : shift * heads])
        return torch.cat(hashes, -1) + hasher.head_row_offsets

    expected = reference()
    expected_history = history.clone()
    history.copy_(before)
    torch.testing.assert_close(hasher.compute_embedding_row_ids(state), expected, atol=0, rtol=0)
    torch.testing.assert_close(history, expected_history, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = hasher.compute_embedding_row_ids(state)
    ids.random_(64)
    history.copy_(before)
    expected = reference()
    expected_history = history.clone()
    history.copy_(before)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(history, expected_history, atol=0, rtol=0)


@pytest.mark.parametrize("vision", [False, True])
def test_engram_restores_cached_tokens_to_shared_and_private_slots(vision):
    from lightllm.models.deepseek_v41.engram import EngramHasher

    hasher = EngramHasher.__new__(EngramHasher)
    hasher.has_vision = vision
    hasher.token_id_to_compressed_id = torch.tensor([1, 0, 2, 1], device="cuda")
    hasher.compressed_token_ids_by_slot = torch.full((12,), -2, dtype=torch.int64, device="cuda")
    request_to_kv_slots = torch.tensor([[3, 6, 8, 0], [3, 6, 10, 0]], dtype=torch.int32, device="cuda")[:, :3]
    token_id = 2 ** 32 + 2 if vision else 2
    hasher.restore_token_history([(0, 1, 0), (0, 2, 1), (1, 1, 0), (1, 2, token_id)], request_to_kv_slots)

    expected = torch.full_like(hasher.compressed_token_ids_by_slot, -2)
    expected[6] = 1
    expected[8] = 0
    expected[10] = -1 if vision else 2
    torch.testing.assert_close(hasher.compressed_token_ids_by_slot, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [0, 1, 385])
def test_gather_rope_tables(rows):
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import gather_rope_tables

    sliding = torch.randn((512, 32), dtype=torch.complex64, device="cuda")
    compress = torch.randn_like(sliding)
    tables = (sliding.real, sliding.imag, compress.real.contiguous(), compress.imag)
    positions = torch.arange(rows * 2, dtype=torch.int32, device="cuda").remainder_(512)[::2]

    def check(outputs):
        for output, table in zip(outputs, tables):
            assert output.is_contiguous()
            torch.testing.assert_close(output, table.index_select(0, positions), atol=0, rtol=0)

    check(gather_rope_tables(positions, tables))
    if rows:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = gather_rope_tables(positions, tables)
        positions.add_(3).remainder_(512)
        graph.replay()
        check(captured)


@pytest.mark.parametrize("byte_offset", [0, 2 ** 31 + 4096, 2 ** 32 + 8192])
def test_candidate_indexer_large_cache_offsets_and_graph(byte_offset):
    from lightllm.models.deepseek_v41.triton_kernel.attention import candidate_indexer_logits

    # V4.1's page table addresses 64-key slabs in 128-byte units across the
    # enclosing MLA pages. A real KV pool can put these slabs beyond 4 GiB.
    storage = torch.empty(byte_offset + 64 * 132, dtype=torch.uint8, device="cuda")
    slab = storage[byte_offset:]
    slab[: 64 * 128].view(torch.float8_e4m3fn).fill_(1)
    slab[64 * 128 :].view(torch.float32).fill_(0.5)
    cache = storage.as_strided((byte_offset // 128 + 1, 64, 1, 132), (128, 132, 132, 1))
    query = torch.ones((2, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.ones((2, 32), device="cuda")
    pages = torch.tensor([[byte_offset // 128], [0]], dtype=torch.int32, device="cuda")
    candidates = torch.tensor([[0], [-1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([8, 0], dtype=torch.int32, device="cuda")

    def run():
        return candidate_indexer_logits(query, cache, weights, pages, candidates, lengths)

    expected = torch.full((2, 8), -float("inf"), device="cuda")
    expected[0].fill_(2048)
    torch.testing.assert_close(run(), expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    weights.fill_(0.25)
    expected[0].fill_(512)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_candidate_selection_2048_blocks_and_recent_block():
    from lightllm.models.deepseek_v41.triton_kernel.attention import select_indexer_candidates

    scores = torch.randn(6, 65536, device="cuda")
    lengths = torch.tensor([0, 7, 16384, 16385, 20000, 60000], dtype=torch.int32, device="cuda")
    indices = torch.empty(6, 2048, dtype=torch.int32, device="cuda")
    select_indexer_candidates(scores, lengths, indices)
    for row, length in enumerate(lengths.tolist()):
        blocks = (length + 7) // 8
        expected = torch.nn.functional.pad(scores[row, :length], (0, -length % 8), value=-float("inf"))
        expected = expected.view(-1, 8).amax(-1)
        if blocks:
            expected[-1] = float("inf")
        chosen = indices[row][indices[row] >= 0].long()
        assert chosen.numel() == min(2048, blocks)
        assert (chosen < blocks).all()
        torch.testing.assert_close(
            expected[chosen].sort(descending=True).values, expected.topk(min(2048, blocks)).values
        )


@pytest.mark.parametrize("row_repeats", [1, 8])
def test_candidate_indexer_fixed_capacity_short_rows_and_updated_graph(row_repeats):
    from lightllm.models.deepseek_v41.triton_kernel.attention import (
        candidate_indexer_logits,
    )

    rows, context, width = 4 * row_repeats, 8192, 2048
    storage = torch.empty((context // 64, 64 * 132), dtype=torch.uint8, device="cuda")
    storage[:, : 64 * 128].view(torch.float8_e4m3fn).fill_(1)
    storage[:, 64 * 128 :].view(torch.float32).fill_(0.5)
    cache = storage.view(-1, 64, 1, 132)
    pages = torch.randperm(context // 64, dtype=torch.int32, device="cuda").expand(rows, -1).contiguous()
    query = torch.ones(rows, 32, 128, dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.ones(rows, 32, device="cuda")
    lengths = torch.tensor([0, 1024, 1081, 8192], dtype=torch.int32, device="cuda").repeat(row_repeats)
    candidates = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    candidates[:, : context // 8] = torch.arange(context // 8, device="cuda")

    def check(actual):
        positions = (candidates[:, :, None] * 8 + torch.arange(8, device="cuda")).flatten(1)
        valid = (candidates[:, :, None] >= 0).expand(-1, -1, 8).flatten(1) & (positions < lengths[:, None])
        expected = torch.where(valid, 2048.0, -float("inf"))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    check(candidate_indexer_logits(query, cache, weights, pages, candidates, lengths))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = candidate_indexer_logits(query, cache, weights, pages, candidates, lengths)
    lengths.copy_(torch.tensor([8192, 0, 1, 1024], device="cuda", dtype=torch.int32).repeat(row_repeats))
    graph.replay()
    check(actual)
