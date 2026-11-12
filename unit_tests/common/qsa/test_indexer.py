import math

import pytest
import torch
import torch.nn.functional as F
import triton

from lightllm.common.qsa.indexer import QSAIndexer
from lightllm.common.qsa.indexer_kernels import _scores


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("minimum", [0, 2])
def test_grouped_scores_keep_chunk_offsets_short_rows_and_mixed_requests(dtype, minimum):
    query = torch.randn(14, 4, 128, device="cuda").to(dtype)
    cache = torch.randn(204, 128, device="cuda").to(dtype)
    table = torch.randperm(204, device="cuda").int().view(3, 68)
    requests = torch.tensor([0] * 9 + [1, 1, 2, 2, 2], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0] * 5 + [0, 3, 8, 16, 7, 9, 12, 17, 11], device="cuda", dtype=torch.int32)
    outputs = [torch.full((9, 17), 42.0, device="cuda") for _ in range(2)]

    def run():
        for group, output in zip((1, 4), outputs):
            block = 64 if group == 1 else 128
            _scores[(triton.cdiv(9, group), 1)](
                query,
                cache,
                requests,
                lengths,
                table,
                output,
                5,
                4,
                128,
                17,
                68,
                True,
                4,
                16,
                128,
                block,
                minimum,
                "ieee" if dtype == torch.float32 else "tf32",
                GROUP_SIZE=group,
                TOKENS=9,
            )

    run()
    torch.testing.assert_close(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)
    if minimum:
        assert torch.all(outputs[1][:2] == 42)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    lengths.zero_()
    query.copy_(torch.randn(query.shape, device="cuda").to(dtype))
    graph.replay()
    torch.testing.assert_close(outputs[1], outputs[0], atol=1e-5, rtol=1e-5)


def make_indexer(dtype=torch.bfloat16, dim=32, heads=4, budget=8, ratio=4):
    projection = torch.randn((heads + 1) * dim, 32, device="cuda", dtype=dtype) / math.sqrt(32)
    return QSAIndexer(
        projection,
        torch.randn(dim, device="cuda", dtype=dtype) * 0.1,
        torch.randn(dim, device="cuda", dtype=dtype) * 0.1,
        heads,
        budget,
        ratio,
    )


def rotary(rows, width, dtype):
    angles = torch.randn(rows, width // 2, device="cuda").repeat(1, 2)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def norm(x, weight, eps):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (normalized * (1 + weight.float())).to(x.dtype)


def rotate(x, cos, sin):
    width = cos.shape[-1]
    first, second = x[..., :width].chunk(2, -1)
    return torch.cat([x[..., :width] * cos + torch.cat([-second, first], -1) * sin, x[..., width:]], -1)


def reference_cache(indexer, raw, table, lengths, cos, sin):
    result = indexer.empty_cache(table.shape[0], table.shape[1]).fill_(float("nan"))
    for row, length in enumerate(lengths):
        count = length // indexer.compress_ratio
        if not count:
            continue
        slots = table[row, : count * indexer.compress_ratio].view(count, indexer.compress_ratio).long()
        pooled = raw[slots].float().mean(1).to(raw.dtype)
        normalized = norm(pooled, indexer.key_norm, indexer.eps)
        result[row, :count] = rotate(normalized, cos[slots[:, 0]], sin[slots[:, 0]])
    return result


def reference_selection(indexer, query, cache, requests, lengths, table):
    rows = []
    for row, (request, length) in enumerate(zip(requests.tolist(), lengths.tolist())):
        count = length // indexer.compress_ratio
        if count:
            scores = (query[row].float() @ cache[request, :count].float().T).relu().sum(0) / math.sqrt(indexer.head_dim)
            blocks = scores.topk(min(count, indexer.token_budget // indexer.compress_ratio)).indices
            positions = (
                blocks[:, None] * indexer.compress_ratio + torch.arange(indexer.compress_ratio, device=query.device)
            ).flatten()
        else:
            positions = torch.empty(0, device=query.device, dtype=torch.int64)
        positions = torch.cat([positions, torch.arange(count * indexer.compress_ratio, length, device=query.device)])
        selected = table[request, positions]
        rows.append(
            F.pad(selected, (0, indexer.token_budget + indexer.compress_ratio - 1 - selected.numel()), value=-1)
        )
    return torch.stack(rows) if rows else table.new_empty((0, indexer.token_budget + indexer.compress_ratio - 1))


def assert_sets(actual, expected):
    torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values, atol=0, rtol=0)


def test_fp8_pool_and_selection():
    torch.manual_seed(1537)
    indexer = make_indexer(dim=128)
    raw = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    cos, sin = rotary(128, 64, raw.dtype)
    table = torch.randperm(128, device="cuda").int().view(2, 64)
    lengths = torch.tensor([64, 52], device="cuda", dtype=torch.int32)
    first = torch.zeros(2, device="cuda", dtype=torch.int32)
    cache = torch.zeros(2, 16, 128, device="cuda", dtype=torch.float8_e4m3fn)
    expected_cache = reference_cache(indexer, raw, table, lengths, cos, sin).to(cache.dtype)
    indexer.pool(raw, table, lengths, cos, sin, cache, first, 16)
    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(
            cache[row, : length // 4].float(), expected_cache[row, : length // 4].float(), rtol=0, atol=0
        )
    query = torch.randn(4, 4, 128, device="cuda").to(cache.dtype)
    requests = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32)
    visible = torch.tensor([64, 41, 52, 3], device="cuda", dtype=torch.int32)
    expected = reference_selection(indexer, query, cache, requests, visible, table)
    assert_sets(indexer.select(query, cache, requests, visible, table), expected)


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("maximum", [0, 3, 8, 41, 64])
def test_prefill_score_bound_preserves_selection_with_short_and_long_rows(physical, dtype, maximum):
    torch.manual_seed(984)
    indexer = make_indexer(dim=128)
    query = torch.randn(9, 4, 128, device="cuda").to(dtype)
    cache = torch.randn((128, 128) if physical else (2, 16, 128), device="cuda").to(dtype)
    table = torch.randperm(128, device="cuda").int().view(2, 64)
    requests = torch.arange(9, device="cuda", dtype=torch.int32) % 2
    lengths = torch.tensor([0, 1, 3, 4, 8, 9, 13, 41, 64], device="cuda", dtype=torch.int32).clamp_max(maximum)
    expected = indexer.select(query, cache, requests, lengths, table, query_chunk_size=3)
    bounded = indexer.select(query, cache, requests, lengths, table, query_chunk_size=3, max_visible_length=maximum)
    assert_sets(bounded, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = indexer.select(query, cache, requests, lengths, table, query_chunk_size=3, max_visible_length=maximum)
    lengths.div_(2, rounding_mode="floor")
    graph.replay()
    assert_sets(actual, indexer.select(query, cache, requests, lengths, table, query_chunk_size=3))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim,rotary_dim,heads,ratio", [(32, 16, 4, 4), (65, 32, 3, 3), (128, 64, 4, 4)])
def test_projection_pooling_and_causal_selection(dtype, dim, rotary_dim, heads, ratio):
    torch.manual_seed(337)
    indexer = make_indexer(dtype, dim, heads, budget=2 * ratio, ratio=ratio)
    hidden = torch.randn(10, 32, device="cuda", dtype=dtype)
    qcos, qsin = rotary(10, rotary_dim, dtype)
    query, raw_new = indexer.project(hidden, qcos, qsin)
    projected = F.linear(hidden, indexer.projection)
    expected_q = projected[:, : heads * dim].reshape(10, heads, dim)
    expected_q = rotate(norm(expected_q, indexer.query_norm, indexer.eps), qcos[:, None], qsin[:, None])
    tolerance = max(2e-6, torch.finfo(dtype).eps * 2)
    torch.testing.assert_close(query, expected_q, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(raw_new, projected[:, heads * dim :], atol=0, rtol=0)
    assert not raw_new.is_contiguous()
    raw_copy = torch.empty_like(raw_new)
    raw_copy.index_copy_(0, torch.arange(raw_new.shape[0], device="cuda"), raw_new)
    torch.testing.assert_close(raw_copy, raw_new.contiguous(), atol=0, rtol=0)
    raw = torch.randn(100, dim, device="cuda", dtype=dtype)
    cos, sin = rotary(100, rotary_dim, dtype)
    table = torch.randperm(100, device="cuda")[:87].view(3, 29).int()
    lengths = [23, 17, 0]
    cache = indexer.empty_cache(3, 29).fill_(float("nan"))
    raw_before = raw.clone()
    indexer.pool(
        raw,
        table,
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        cos,
        sin,
        cache,
        torch.zeros(3, device="cuda", dtype=torch.int32),
        29 // ratio,
    )
    expected_cache = reference_cache(indexer, raw, table, lengths, cos, sin)
    torch.testing.assert_close(cache, expected_cache, atol=tolerance, rtol=tolerance, equal_nan=True)
    torch.testing.assert_close(raw, raw_before, atol=0, rtol=0)
    requests = torch.tensor([0, 0, 1, 0, 1, 1, 0, 1, 1, 0], device="cuda", dtype=torch.int32)
    visible = torch.tensor([0, 1, 3, 4, 5, 8, 9, 15, 17, 23], device="cuda", dtype=torch.int32)
    actual = indexer.select(query, cache, requests, visible, table, query_chunk_size=3)

    expected = reference_selection(indexer, query, cache, requests, visible, table)
    assert_sets(actual, expected)
    for row, (request, length) in enumerate(zip(requests.tolist(), visible.tolist())):
        allowed = set(table[request, :length].tolist())
        assert set(actual[row][actual[row] >= 0].tolist()) <= allowed


def test_incremental_pool_cache_and_cuda_graph_replay():
    torch.manual_seed(337)
    indexer = make_indexer()
    raw = torch.randn(48, 32, device="cuda", dtype=torch.bfloat16)
    cos, sin = rotary(48, 16, raw.dtype)
    table = torch.randperm(48, device="cuda")[:40].view(2, 20).int()
    cache = indexer.empty_cache(2, 20).fill_(float("nan"))
    lengths = torch.tensor([7, 5], device="cuda", dtype=torch.int32)
    first = torch.zeros_like(lengths)
    indexer.pool(raw, table, lengths, cos, sin, cache, first, 2)
    original = cache[:, :1].clone()
    hidden = torch.randn(2, 32, device="cuda", dtype=raw.dtype)
    qcos, qsin = rotary(2, 16, raw.dtype)
    requests = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    first.fill_(1)
    lengths.copy_(torch.tensor([13, 10], device="cuda", dtype=torch.int32))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            query, _ = indexer.project(hidden, qcos, qsin)
            indexer.pool(raw, table, lengths, cos, sin, cache, first, 2)
            indexer.select(query, cache, requests, lengths, table, query_chunk_size=1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query, _ = indexer.project(hidden, qcos, qsin)
        indexer.pool(raw, table, lengths, cos, sin, cache, first, 2)
        selected = indexer.select(query, cache, requests, lengths, table, query_chunk_size=1)
    for counts, starts in [([13, 10], [1, 1]), ([18, 15], [3, 2])]:
        hidden.normal_()
        lengths.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        first.copy_(torch.tensor(starts, device="cuda", dtype=torch.int32))
        graph.replay()
        expected_cache = reference_cache(indexer, raw, table, counts, cos, sin)
        tolerance = 2 * torch.finfo(raw.dtype).eps
        torch.testing.assert_close(cache, expected_cache, atol=tolerance, rtol=tolerance, equal_nan=True)
        assert_sets(selected, reference_selection(indexer, query, cache, requests, lengths, table))
        torch.testing.assert_close(cache[:, :1], original, atol=0, rtol=0)


def test_zero_score_ties_tail_only_empty_and_checkpoint_binding():
    indexer = make_indexer()
    prefix = "model.language_model.layers.3.self_attn.indexer"
    weights = {
        f"{prefix}.index_qk_proj.weight": indexer.projection,
        f"{prefix}.q_layernorm.weight": indexer.query_norm,
        f"{prefix}.k_layernorm.weight": indexer.key_norm,
    }
    bound = QSAIndexer.from_checkpoint(weights, prefix, token_budget=8)
    assert bound.projection is indexer.projection
    query = torch.zeros(3, 4, 32, device="cuda", dtype=torch.bfloat16)
    table = torch.arange(29, device="cuda", dtype=torch.int32).view(1, -1)
    cache = torch.randn(1, 7, 32, device="cuda", dtype=query.dtype)
    requests = torch.zeros(3, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 3, 21], device="cuda", dtype=torch.int32)
    selected = bound.select(query, cache, requests, lengths, table)
    assert torch.all(selected[0] == -1)
    assert set(selected[1][selected[1] >= 0].tolist()) == {0, 1, 2}
    chosen = selected[2][selected[2] >= 0].tolist()
    assert len(chosen) == len(set(chosen)) == 9
    assert chosen[-1] == 20
    for offset in [0, 4]:
        block = chosen[offset : offset + 4]
        assert block == list(range(block[0], block[0] + 4)) and block[0] % 4 == 0
    empty_query, raw = bound.project(query.new_empty((0, 32)), query.new_empty((0, 0)), query.new_empty((0, 0)))
    assert raw.shape == (0, 32)
    assert bound.select(empty_query, cache, requests[:0], lengths[:0], table).shape == (0, 11)

    short = bound.select(query[:1], cache[:, :0].contiguous(), requests[:1], lengths[1:2], table[:, :3].contiguous())
    assert set(short[0][short[0] >= 0].tolist()) == {0, 1, 2}


def test_invalid_layouts():
    indexer = make_indexer()
    with pytest.raises(ValueError, match="budget"):
        QSAIndexer(indexer.projection, indexer.query_norm, indexer.key_norm, token_budget=3)
    with pytest.raises(ValueError, match="rotary"):
        indexer.project(
            indexer.projection.new_empty((2, 32)),
            indexer.projection.new_empty((2, 3)),
            indexer.projection.new_empty((2, 3)),
        )
    with pytest.raises(ValueError, match="int32"):
        indexer.select(
            indexer.projection.new_empty((2, 4, 32)),
            indexer.empty_cache(1, 4),
            torch.zeros(2, device="cuda", dtype=torch.int64),
            torch.ones(2, device="cuda", dtype=torch.int32),
            torch.zeros(1, 4, device="cuda", dtype=torch.int32),
        )


def test_rejected_suffix_repool_overwrites_candidate_blocks():
    torch.manual_seed(337)
    indexer = make_indexer()
    table = torch.arange(20, device="cuda", dtype=torch.int32).view(1, -1)
    raw = torch.randn(20, 32, device="cuda", dtype=torch.bfloat16)
    cos, sin = rotary(20, 16, raw.dtype)
    cache = indexer.empty_cache(1, 20).fill_(float("nan"))
    indexer.pool(
        raw,
        table,
        torch.tensor([13], device="cuda", dtype=torch.int32),
        cos,
        sin,
        cache,
        torch.tensor([0], device="cuda", dtype=torch.int32),
        3,
    )
    prefix = cache[:, :1].clone()

    raw[7:12].normal_()
    lengths = torch.tensor([12], device="cuda", dtype=torch.int32)
    indexer.pool(raw, table, lengths, cos, sin, cache, torch.tensor([7 // 4], device="cuda", dtype=torch.int32), 2)
    expected = reference_cache(indexer, raw, table, [12], cos, sin)
    torch.testing.assert_close(cache, expected, atol=0.016, rtol=0.016, equal_nan=True)
    torch.testing.assert_close(cache[:, :1], prefix, atol=0, rtol=0)
    query = torch.randn(1, 4, 32, device="cuda", dtype=raw.dtype)
    requests = torch.zeros(1, device="cuda", dtype=torch.int32)
    assert_sets(
        indexer.select(query, cache, requests, lengths, table),
        reference_selection(indexer, query, cache, requests, lengths, table),
    )


def test_million_token_context_with_bounded_score_chunks():
    torch.manual_seed(337)
    context = 1048579
    indexer = make_indexer(dim=128, budget=2048)
    cache = torch.randn(1, context // 4, 128, device="cuda", dtype=torch.bfloat16)

    table = torch.arange(context - 1, -1, -1, device="cuda", dtype=torch.int32).view(1, -1)
    query = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    requests = torch.zeros(3, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([4099, 262147, context], device="cuda", dtype=torch.int32)
    actual = indexer.select(query, cache, requests, lengths, table, query_chunk_size=1)
    expected = reference_selection(indexer, query, cache, requests, lengths, table)
    assert_sets(actual, expected)


def test_active_rows_pool_into_shared_request_table():
    torch.manual_seed(337)
    indexer = make_indexer()
    table = torch.randperm(80, device="cuda").view(4, 20).int()
    raw = torch.randn(80, 32, device="cuda", dtype=torch.bfloat16)
    cos, sin = rotary(80, 16, raw.dtype)
    cache = indexer.empty_cache(4, 20).fill_(float("nan"))
    rows = torch.tensor([3, -1, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([13, 0, 9], device="cuda", dtype=torch.int32)
    indexer.pool(raw, table, lengths, cos, sin, cache, torch.zeros_like(lengths), 4, request_rows=rows)
    expected = reference_cache(indexer, raw, table, [0, 9, 0, 13], cos, sin)
    torch.testing.assert_close(cache, expected, atol=0.016, rtol=0.016, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("queries_per_request", [1, 4])
@pytest.mark.parametrize("query_chunk_size", [3, 64])
def test_physical_cache_partial_prefix_forks_and_graph_replay(dtype, queries_per_request, query_chunk_size):
    torch.manual_seed(740)
    indexer = make_indexer(dtype)
    raw = torch.randn(32, 32, device="cuda", dtype=dtype)
    cos, sin = rotary(32, 16, dtype)

    table = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12], [0, 1, 2, 3, 4, 5, 6, 8, 13, 14, 15, 16]],
        device="cuda",
        dtype=torch.int32,
    )
    cache = indexer.empty_physical_cache(raw.shape[0]).fill_(float("nan"))
    lengths = torch.tensor([7, 7], device="cuda", dtype=torch.int32)
    first = torch.zeros_like(lengths)
    rows = torch.tensor([1, -1, 0], device="cuda", dtype=torch.int32)
    active_lengths = torch.tensor([7, 0, 7], device="cuda", dtype=torch.int32)
    active_first = torch.zeros_like(active_lengths)
    query = torch.randn(2 * queries_per_request, indexer.query_heads, indexer.head_dim, device="cuda", dtype=dtype)
    requests = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(queries_per_request)

    def run():
        indexer.pool(raw, table, active_lengths, cos, sin, cache, active_first, 3, request_rows=rows)
        return indexer.select(
            query,
            cache,
            requests,
            lengths.repeat_interleave(queries_per_request),
            table,
            query_chunk_size=query_chunk_size,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = run()

    for length in (8, 12, 5, 12):
        lengths.fill_(length)
        active_lengths.copy_(torch.tensor([length, 0, length], device="cuda", dtype=torch.int32))
        graph.replay()
        dense = indexer.empty_cache(2, table.shape[1]).fill_(float("nan"))
        indexer.pool(raw, table, lengths, cos, sin, dense, first, 3)
        for request in range(2):
            count = length // indexer.compress_ratio
            end_slots = table[
                request, indexer.compress_ratio - 1 : count * indexer.compress_ratio : indexer.compress_ratio
            ]
            torch.testing.assert_close(cache[end_slots.long()], dense[request, :count], atol=0, rtol=0)
        assert_sets(
            selected, indexer.select(query, dense, requests, lengths.repeat_interleave(queries_per_request), table)
        )
    assert not torch.equal(cache[7], cache[8])

    untouched = cache[31].clone()
    raw[7].add_(3)
    run()
    dense = indexer.empty_cache(2, table.shape[1])
    indexer.pool(raw, table, lengths, cos, sin, dense, first, 3)
    torch.testing.assert_close(cache[7], dense[0, 1], atol=0, rtol=0)
    torch.testing.assert_close(cache[31], untouched, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("capacity", [4099, 262159])
def test_sparse_visibility_in_large_cache_and_dynamic_graph(dtype, capacity):
    torch.manual_seed(38413)
    indexer = make_indexer(dtype, dim=128, heads=4, budget=8)
    query = torch.randn(8, 4, 128, device="cuda", dtype=dtype)
    table = torch.stack([torch.randperm(capacity, device="cuda") for _ in range(3)]).int()
    requests = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], device="cuda", dtype=torch.int32)
    original = torch.tensor([0, 3, 4, 257, 255, 256, capacity - 1, capacity], device="cuda", dtype=torch.int32)
    lengths = original.clone()
    cache = torch.randn(3, capacity // 4, 128, device="cuda", dtype=dtype)
    cache[0, 257 // 4 :] = float("nan")
    cache[1, 256 // 4 :] = float("nan")
    actual = indexer.select(query, cache, requests, lengths, table)
    assert_sets(actual, reference_selection(indexer, query, cache, requests, lengths, table))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer.select(query, cache, requests, lengths, table)
    for pattern in (torch.zeros_like(original), original):
        lengths.copy_(pattern)
        graph.replay()
        assert_sets(output, reference_selection(indexer, query, cache, requests, lengths, table))
    graph.reset()


@pytest.mark.parametrize("rows", [4, 36, 40, 64, 128])
def test_indexer_preprojected_strided_graph(rows):
    torch.manual_seed(7912)
    projection = torch.randn(640, 2560, dtype=torch.bfloat16, device="cuda") * 0.02
    indexer = QSAIndexer(
        projection,
        torch.randn(128, dtype=torch.bfloat16, device="cuda") * 0.1,
        torch.randn(128, dtype=torch.bfloat16, device="cuda") * 0.1,
    )
    hidden = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
    angles = torch.randn(rows, 32, device="cuda").repeat(1, 2)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    packed = torch.empty(rows, 2560, dtype=hidden.dtype, device="cuda")
    projected = packed[:, 1920:]
    assert not projected.is_contiguous()
    projected.copy_(F.linear(hidden, projection))
    indexer.project(hidden, cos, sin, projected=projected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = indexer.project(hidden, cos, sin, projected=projected)
    for _ in range(8):
        hidden.normal_()
        expected = indexer.project(hidden, cos, sin)
        projected.copy_(F.linear(hidden, projection))
        hidden.zero_()
        graph.replay()
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
    graph.reset()
