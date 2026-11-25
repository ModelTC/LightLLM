from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v41.triton_kernel.attention import pool_and_norm_pairs, select_indexer_topk


@pytest.mark.parametrize("split", [1, 3, 5])
def test_c2_carries_partial_groups_across_chunk_boundaries(split):
    kv, scores = torch.randn(7, 1024, device="cuda").split(512, -1)
    positions = torch.arange(7, dtype=torch.int32, device="cuda")
    req = torch.zeros(7, dtype=torch.int32, device="cuda")
    state = torch.zeros(2, 4, 2, 512, device="cuda")
    weight = torch.rand(512, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6
    first = pool_and_norm_pairs(kv[:split], scores[:split], req[:split], positions[:split], state, 1, weight, eps)
    second = pool_and_norm_pairs(kv[split:], scores[split:], req[split:], positions[split:], state, 1, weight, eps)
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
    out = pool_and_norm_pairs(kv, scores, req, pos, state, 2, weight, eps)
    expected = previous[1, 0, 0] + (kv[2] - previous[1, 0, 0]) * torch.sigmoid(scores[2] - previous[1, 0, 1])
    torch.testing.assert_close(
        out[2], rmsnorm_forward(expected.bfloat16()[None], weight, eps)[0], atol=0.015625, rtol=0.008
    )
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
    select_indexer_topk(logits, valid, pages, slots, lengths, 8, mask, 2)
    assert mask.tolist() == [[False, True, False, True]]
    assert slots.tolist() == [[43, 44, 45, 46, 47]]
    deeper_scores = torch.arange(32, 0, -1, device="cuda", dtype=torch.float32).view(1, -1)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [16, 5120])
@pytest.mark.parametrize("masked", [False, True])
def test_engram_gate_matches_reference_and_cuda_graph(dtype, dim, masked):
    from lightllm.models.deepseek_v41.engram import engram_forward

    tokens, mult = 7, 4
    # Both producers may expose row slices; no contiguous copies are needed.
    streams = torch.randn(tokens * 2, mult, dim, device="cuda", dtype=dtype)[::2]
    kv = torch.randn(tokens * 2, (mult + 1) * dim, device="cuda", dtype=dtype)[::2]
    streams[0].zero_()
    kv[0, : mult * dim].zero_()
    weight = SimpleNamespace(
        engram_embed_=SimpleNamespace(lookup=lambda _, group: None),
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

    check(engram_forward(streams, None, weight, eps, None, token_mask))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = engram_forward(streams, None, weight, eps, None, token_mask)
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
        select_indexer_topk(logits, lengths, pages, slots, counts, 8, mask, candidate_topk)

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
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_norm_rope

    tokens = 67
    storage = torch.randn(tokens, padded, 512, device="cuda", dtype=torch.bfloat16)
    original = storage.clone()
    active = storage[:, :heads]
    expected = torch.empty_like(active)
    pos = torch.randperm(tokens, device="cuda", dtype=torch.int32)
    freqs = torch.polar(torch.ones(tokens, 32, device="cuda"), torch.randn(tokens, 32, device="cuda"))
    fused_q_norm_rope(original[:, :heads], expected, 1e-6, freqs, pos, apply_norm=False)
    fused_q_norm_rope(active, active, 1e-6, freqs, pos, apply_norm=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_q_norm_rope(active, active, 1e-6, freqs, pos, apply_norm=False)
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
    monkeypatch.setattr(engram, "build_compressed_token_map", lambda _: (list(range(64)), 64))
    length = tokens + 132
    table = torch.randperm(3 * length, device="cuda").int().view(3, length)
    hasher = engram.EngramHasher(config, None, table.numel())
    hasher.history.random_(64)
    hasher.history[table[:, 126].long()] = -1 if vision else 0
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
    before = hasher.history.clone()

    def reference():
        hasher.history[state.mem_index.long()] = hasher._map_token_ids(ids)
        positions, reqs = position.long(), req.long()
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        previous_tokens = []
        for shift in range(hasher.order):
            slot = table[reqs, (positions - shift).clamp_min(0)].long()
            source = hasher.history[slot]
            blocked |= (positions < shift) | (source == -1)
            previous_tokens.append(torch.where(blocked, hasher.pad, source))
        products = torch.stack(previous_tokens, -1)[:, None, :] * hasher.multipliers[None, :, :]
        rolling, hashes = products[..., 0], []
        for shift in range(1, hasher.order):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling[..., None] % hasher.primes[:, (shift - 1) * heads : shift * heads])
        return torch.cat(hashes, -1) + hasher.offsets

    expected = reference()
    expected_history = hasher.history.clone()
    hasher.history.copy_(before)
    torch.testing.assert_close(hasher(state), expected, atol=0, rtol=0)
    torch.testing.assert_close(hasher.history, expected_history, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = hasher(state)
    ids.random_(64)
    hasher.history.copy_(before)
    expected = reference()
    expected_history = hasher.history.clone()
    hasher.history.copy_(before)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(hasher.history, expected_history, atol=0, rtol=0)


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
