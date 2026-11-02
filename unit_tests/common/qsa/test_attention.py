import pytest
import torch
import torch.nn.functional as F

from lightllm.common.qsa.attention import indexed_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def reference(query, key, value, indices, scale=None):
    selected_k = key[indices.clamp_min(0)].transpose(1, 2)
    selected_v = value[indices.clamp_min(0)].transpose(1, 2)
    mask = (indices >= 0)[:, None, None, :]
    return F.scaled_dot_product_attention(
        query.unsqueeze(2), selected_k, selected_v, attn_mask=mask, scale=scale, enable_gqa=True
    ).squeeze(2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rows,qheads,kvheads,dim,vdim,count", [(3, 6, 2, 32, 16, 79), (1, 24, 2, 256, 256, 2051), (5, 3, 1, 65, 33, 11)]
)
@pytest.mark.parametrize("splits", [1, 4, None])
@pytest.mark.parametrize("bounded", [False, True])
def test_sparse_gqa_matches_sdpa(dtype, rows, qheads, kvheads, dim, vdim, count, splits, bounded):
    torch.manual_seed(437)
    slots = count + 31
    q = torch.randn(rows, qheads, dim, device="cuda", dtype=dtype)
    k = torch.randn(slots, kvheads, dim, device="cuda", dtype=dtype)
    v = torch.randn(slots, kvheads, vdim, device="cuda", dtype=dtype)
    indices = torch.stack([torch.randperm(slots, device="cuda")[:count] for _ in range(rows)]).int()
    indices[:, ::7] = -1
    lengths = torch.full((rows,), count + 17, device="cuda", dtype=torch.int32) if bounded else None
    if bounded:
        indices[0, count // 2 :] = -1
        lengths[0] = count // 2
    if rows > 1:
        indices[-1].fill_(-1)
        if bounded:
            lengths[-1] = -1
    before = (q.clone(), k.clone(), v.clone(), indices.clone())
    actual = indexed_attention(q, k, v, indices, splits=splits, selection_lengths=lengths)
    expected = reference(q, k, v, indices)
    tolerance = 2e-5 if dtype == torch.float32 else torch.finfo(dtype).eps * 2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for tensor, original in zip((q, k, v, indices), before):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)
    if rows > 1:
        assert torch.count_nonzero(actual[-1]) == 0


@pytest.mark.parametrize("bounded", [False, True])
def test_attention_cuda_graph_replay_and_custom_scale(bounded):
    q = torch.randn(2, 6, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(128, 2, 32, device="cuda", dtype=q.dtype)
    v = torch.randn(128, 2, 16, device="cuda", dtype=q.dtype)
    indices = torch.arange(79, device="cuda", dtype=torch.int32).expand(2, -1).contiguous()
    lengths = torch.full((2,), 79, device="cuda", dtype=torch.int32) if bounded else None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            indexed_attention(q, k, v, indices, scale=0.17, selection_lengths=lengths)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexed_attention(q, k, v, indices, scale=0.17, selection_lengths=lengths)
    for bounds in ([13, 0], [128, 63]):
        q.normal_()
        indices.copy_(torch.stack([torch.randperm(128, device="cuda")[:79] for _ in range(2)]).int())
        indices[:, ::3] = -1
        if bounded:
            lengths.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
            indices.masked_fill_(torch.arange(79, device="cuda")[None, :] >= lengths[:, None], -1)
        graph.replay()
        torch.testing.assert_close(output, reference(q, k, v, indices, scale=0.17), atol=0.016, rtol=0.016)


def test_empty_selection_and_invalid_shapes():
    q = torch.randn(2, 6, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(10, 2, 32, device="cuda", dtype=q.dtype)
    v = torch.randn(10, 2, 16, device="cuda", dtype=q.dtype)
    ids = torch.empty(2, 0, device="cuda", dtype=torch.int32)
    assert torch.count_nonzero(indexed_attention(q, k, v, ids)) == 0
    assert indexed_attention(q[:0], k, v, ids[:0]).shape == (0, 6, 16)
    with pytest.raises(ValueError, match="divisible"):
        indexed_attention(q[:, :5].contiguous(), k, v, ids)
    with pytest.raises(ValueError, match="int32"):
        indexed_attention(q, k, v, ids.long())
    with pytest.raises(ValueError, match="Selection lengths"):
        indexed_attention(q, k, v, ids, selection_lengths=torch.zeros(2, device="cuda", dtype=torch.int64))


def test_native_interleaved_kv_views_match_contiguous_cache():
    q = torch.randn(3, 24, 256, device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(97, 4, 256, device="cuda", dtype=q.dtype)
    k, v = packed[:, :2], packed[:, 2:]
    assert not k.is_contiguous() and not v.is_contiguous()
    indices = torch.stack([torch.randperm(97, device="cuda")[:79] for _ in range(3)]).int()
    indices[:, ::7] = -1
    actual = indexed_attention(q, k, v, indices)
    expected = indexed_attention(q, k.contiguous(), v.contiguous(), indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_direct_short_rows_preserve_fixed_partition_and_graph_replay(dtype):
    torch.manual_seed(701)
    rows, slots, width = 5, 4096, 2051
    q = torch.randn(rows, 6, 256, device="cuda", dtype=dtype)
    k = torch.randn(slots, 1, 256, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    ids = torch.stack([torch.randperm(slots, device="cuda")[:width] for _ in range(rows)]).int()
    lengths = torch.tensor([0, 16, 64, 65, 2048], device="cuda", dtype=torch.int32)

    def compute(direct):
        return indexed_attention(q, k, v, ids, splits=32, selection_lengths=lengths, short_rows_direct=direct)

    torch.testing.assert_close(compute(True), compute(False), atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compute(True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compute(True)
    for bounds in ([2048, 65, 64, 0, 16], [64, 2048, 0, 65, 63]):
        lengths.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
        q.normal_()
        graph.replay()
        torch.testing.assert_close(actual, compute(False), atol=0, rtol=0)
