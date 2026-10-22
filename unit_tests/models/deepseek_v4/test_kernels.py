import pytest
import torch

from lightllm.models.deepseek_v4.triton_kernel import norm_rope, topk_softplus_sqrt, topk_transform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def freqs_and_positions(batch, dtype=torch.int64):
    torch.manual_seed(13)
    angles = torch.randn(128, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(0, 128, (batch,), device="cuda", dtype=dtype)
    return freqs, positions


def rope_reference(x, freqs, positions):
    pairs = torch.view_as_complex(x[..., -64:].float().reshape(*x.shape[:-1], 32, 2).contiguous())
    rotation = freqs[positions.long()]
    while rotation.ndim < pairs.ndim:
        rotation = rotation.unsqueeze(1)
    return torch.view_as_real(pairs * rotation).flatten(-2)


@pytest.mark.parametrize("batch,heads", [(0, 8), (1, 8), (7, 16), (128, 16)])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_q_norm_rope_strided_and_bf16_rounding(batch, heads, position_dtype):
    freqs, pos = freqs_and_positions(batch, position_dtype)
    q = torch.randn(batch, heads + 2, 512, device="cuda", dtype=torch.bfloat16)[:, :heads]
    padded = torch.full((batch, heads + 4, 512), 9, device="cuda", dtype=torch.bfloat16)
    out = padded[:, :heads]
    norm_rope.fused_q_norm_rope(q, out, 1e-6, freqs, pos)
    normalized = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    expected = normalized.clone()
    expected[..., -64:] = rope_reference(normalized, freqs, pos).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.02)
    assert (padded[:, heads:] == 9).all()
    # In-place outputs must also work without changing the normalization input early.
    inplace = q.contiguous().clone()
    norm_rope.fused_q_norm_rope(inplace, inplace, 1e-6, freqs, pos)
    torch.testing.assert_close(inplace, expected, rtol=0.01, atol=0.02)


@pytest.mark.parametrize("batch", [1, 7, 129])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_k_norm_rope_cache_layout_and_unwritten_bytes(batch, position_dtype):
    freqs, pos = freqs_and_positions(batch, position_dtype)
    kv = torch.randn(batch, 640, device="cuda", dtype=torch.bfloat16)[:, :512]
    kv[0].zero_()
    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(384, device="cuda")[:batch].int()
    cache = torch.full((3, 74880), 165, device="cuda", dtype=torch.uint8)
    norm_rope.fused_k_norm_rope_flashmla(kv, weight, 1e-6, freqs, pos, slots, cache, 128)
    values = kv.float() * torch.rsqrt(kv.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    grouped = values[:, :448].reshape(batch, 7, 64)
    scale_raw = grouped.abs().amax(-1).clamp_min(1e-4) / 448
    bits = scale_raw.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    scales = torch.ldexp(torch.ones_like(scale_raw), exponent - 127)
    expected_fp8 = (grouped / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected_rope = rope_reference(values, freqs, pos).bfloat16()
    pages = slots.long() // 128
    offsets = slots.long() % 128
    data_offsets = offsets[:, None] * 576 + torch.arange(576, device="cuda")
    data = cache[pages[:, None], data_offsets]
    torch.testing.assert_close(
        data[:, :448].contiguous().view(torch.float8_e4m3fn).float(),
        expected_fp8.reshape(batch, 448).float(),
        rtol=0.13,
        atol=0.03125,
    )
    torch.testing.assert_close(data[:, 448:].contiguous().view(torch.bfloat16), expected_rope, rtol=0.01, atol=0.02)
    scale_offsets = 128 * 576 + offsets[:, None] * 8 + torch.arange(7, device="cuda")
    torch.testing.assert_close(cache[pages[:, None], scale_offsets], exponent.byte(), rtol=0, atol=0)
    written = torch.zeros_like(cache, dtype=torch.bool)
    written[pages[:, None], data_offsets] = True
    written[pages[:, None], scale_offsets] = True
    assert (cache[~written] == 165).all()


@pytest.mark.parametrize("batch,heads", [(0, 64), (1, 3), (7, 64), (128, 64)])
def test_indexer_rope_hadamard_quant_and_preallocated_outputs(batch, heads):
    freqs, pos = freqs_and_positions(batch)
    q = torch.randn(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16)
    if batch:
        q[0, 0].zero_()
    output = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    out_weights = torch.empty(batch, heads, 1, device="cuda")
    got, got_weights = norm_rope.fused_q_indexer_rope_quant(q, weights, 0.01, freqs, pos, output, out_weights)
    assert got is output and got_weights is out_weights
    rotated = q.float()
    rotated[..., -64:] = rope_reference(q, freqs, pos)
    hadamard = torch.ones(1, 1, device="cuda")
    for _ in range(7):
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1), torch.cat((hadamard, -hadamard), 1)), 0)
    transformed = (rotated @ hadamard) * (128 ** -0.5)
    scale = transformed.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    expected = (transformed / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(got.float(), expected.float(), rtol=0.13, atol=0.03125)
    torch.testing.assert_close(got_weights, weights.float().unsqueeze(-1) * 0.01 * scale, rtol=4e-6, atol=1e-8)


def router_reference(logits, factor, bias, tokens, table, image_bias, image_start, top_k=6):
    scores = torch.sqrt(logits.clamp_min(0) + torch.log(1 + torch.exp(-logits.abs())))
    scores = torch.nan_to_num(scores, nan=0)
    selection = scores + (bias if bias is not None else 0)
    if image_bias is not None:
        image = tokens >= image_start
        selection = torch.where(image[:, None], scores + image_bias, selection)
    indices = selection.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
    if table is not None:
        hash_ids = table[tokens.clamp_max(table.shape[0] - 1)]
        if image_bias is None:
            indices = hash_ids
        else:
            indices = torch.where(image[:, None], indices, hash_ids)
    weights = scores.gather(1, indices)
    total = weights.sum(-1, keepdim=True)
    return weights * factor / torch.where(total > 0, total, 1), indices


@pytest.mark.parametrize("experts", [256, 384])
@pytest.mark.parametrize("mode", ["bias", "hash", "vision", "hash_vision"])
@pytest.mark.parametrize("rows", [1, 17, 513])
@pytest.mark.parametrize("distribution", ["normal", "ties", "underflow"])
def test_router_modes_scaling_ties_and_underflow(experts, mode, rows, distribution):
    torch.manual_seed(23)
    logits = torch.randn(rows, experts, device="cuda")
    bias = torch.randn(experts, device="cuda") * 0.1
    image_bias = torch.randn_like(bias) * 0.1 if "vision" in mode else None
    if distribution == "ties":
        logits.zero_()
        bias.zero_()
        if image_bias is not None:
            image_bias.zero_()
    elif distribution == "underflow":
        logits.fill_(-100)
    tokens = torch.randint(0, 32, (rows,), device="cuda")
    table = torch.stack([torch.randperm(experts, device="cuda")[:6] for _ in range(32)]) if "hash" in mode else None
    if image_bias is not None:
        tokens[::2] = 100_000 + torch.arange(0, rows, 2, device="cuda")
    correction = None if table is not None else bias
    weights = torch.empty(rows, 6, device="cuda")
    indices = torch.empty(rows, 6, device="cuda", dtype=torch.int64)
    topk_softplus_sqrt.topk_softplus_sqrt(weights, indices, logits, 2.5, correction, tokens, table, image_bias, 32)
    ref_weights, ref_indices = router_reference(logits, 2.5, correction, tokens, table, image_bias, 32)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(weights, ref_weights, rtol=5e-6, atol=1e-6)


@pytest.mark.parametrize("top_k", [2, 3])
def test_router_draft_topk_respects_output_bounds(top_k):
    rows, experts = 160, 128
    logits = torch.randn(rows, experts, device="cuda")
    bias = torch.randn(experts, device="cuda")
    # Keep the old six-expert writer inside the allocation so an overwrite fails deterministically.
    weight_storage = torch.full((rows * 6 + 8,), -1.0, device="cuda")
    index_storage = torch.full((rows * 6 + 8,), -1, dtype=torch.int64, device="cuda")
    weights = weight_storage[4 : 4 + rows * top_k].view(rows, top_k)
    indices = index_storage[4 : 4 + rows * top_k].view(rows, top_k)
    topk_softplus_sqrt.topk_softplus_sqrt(weights, indices, logits, 2.5, correction_bias=bias)
    for storage in (weight_storage, index_storage):
        assert (storage[:4] == -1).all() and (storage[4 + rows * top_k :] == -1).all()
    ref_weights, ref_indices = router_reference(logits, 2.5, bias, None, None, None, 0, top_k=top_k)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(weights, ref_weights, rtol=5e-6, atol=1e-6)


def assert_topk(scores, lengths, pages, out, raw, page_size):
    for row, length in enumerate(lengths.tolist()):
        count = min(out.shape[1], length)
        chosen = raw[row, :count].long()
        assert (raw[row, count:] == -1).all() and (out[row, count:] == -1).all()
        assert chosen.unique().numel() == count
        assert ((chosen >= 0) & (chosen < length)).all()
        torch.testing.assert_close(
            out[row, :count], (pages[row, chosen // page_size] * page_size + chosen % page_size).int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scores[row, chosen].sort(descending=True).values, scores[row, :length].topk(count).values, rtol=0, atol=0
        )


@pytest.mark.parametrize("width", [128, 512, 640, 8192, 262144])
@pytest.mark.parametrize("page_size", [1, 64, 128])
def test_topk_valid_prefix_raw_indices_and_padding(width, page_size):
    torch.manual_seed(31)
    scores = torch.randn(8, width + 128, device="cuda")[:, :width]
    lengths = torch.tensor(
        [0, 1, min(511, width), min(512, width), min(513, width), width // 3, width - 3, width],
        device="cuda",
        dtype=torch.int32,
    )
    scores.masked_fill_(torch.arange(width, device="cuda")[None, :] >= lengths[:, None], float("nan"))
    pages = torch.randint(0, 4096, (8, (width + page_size - 1) // page_size), device="cuda", dtype=torch.int32)
    out = torch.empty(8, 512, device="cuda", dtype=torch.int32)
    raw = torch.empty_like(out)
    topk_transform.topk_transform(scores, lengths, pages, out, page_size, raw)
    assert_topk(scores, lengths, pages, out, raw, page_size)
    # The allocation-free in-place translation must select the same values too.
    topk_transform.topk_transform(scores, lengths, pages, out, page_size)
    valid = out >= 0
    assert valid.sum(-1).tolist() == lengths.clamp_max(512).tolist()


@pytest.mark.parametrize("width,rows", [(513, 1), (640, 2), (1024, 1), (1024, 4), (1536, 2), (2048, 2)])
@pytest.mark.parametrize("distribution", ["normal", "ties", "concentrated"])
@pytest.mark.parametrize("page_size", [1, 64, 128])
def test_short_topk_rank_selection_and_optional_raw(width, rows, distribution, page_size):
    torch.manual_seed(43)
    scores = torch.randn(rows, width + 1, device="cuda")[:, :width]
    if distribution == "ties":
        scores.fill_(1)
    elif distribution == "concentrated":
        scores.mul_(1e-6).add_(1)
    lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    if rows > 1:
        lengths[0] = 513
    scores.masked_fill_(torch.arange(width, device="cuda")[None, :] >= lengths[:, None], float("nan"))
    pages = torch.randint(0, 4096, (rows, (width + page_size - 1) // page_size), device="cuda", dtype=torch.int32)
    out = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    raw = torch.empty_like(out)
    topk_transform.topk_transform(scores, lengths, pages, out, page_size, raw)
    assert_topk(scores, lengths, pages, out, raw, page_size)
    expected = out.clone()
    topk_transform.topk_transform(scores, lengths, pages, out, page_size)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows,width", [(8, 8192), (1, 1024), (2, 2048), (4, 1024)])
def test_topk_cuda_graph_replay_changing_lengths(rows, width):
    scores = torch.randn(rows, width, device="cuda")
    lengths = torch.zeros(rows, device="cuda", dtype=torch.int32)
    pages = torch.randint(0, 4096, (rows, width // 64), device="cuda", dtype=torch.int32)
    out = torch.empty(rows, 512, device="cuda", dtype=torch.int32)
    raw = torch.empty_like(out)
    fn = lambda: topk_transform.topk_transform(scores, lengths, pages, out, 64, raw)
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for shift in range(8):
        scores.normal_()
        lengths.copy_(
            torch.tensor([0, 1, 511, 512, 513, 1024, 4096, width], device="cuda", dtype=torch.int32)
            .clamp_max(width)
            .roll(shift)[:rows]
        )
        scores.masked_fill_(torch.arange(width, device="cuda")[None, :] >= lengths[:, None], float("nan"))
        graph.replay()
        assert_topk(scores, lengths, pages, out, raw, 64)


def test_norm_cuda_graph_dependent_launch_and_replay():
    freqs, positions = freqs_and_positions(7)
    source = torch.randn(7, 8, 512, device="cuda", dtype=torch.bfloat16)
    first = torch.empty_like(source)
    second = torch.empty_like(source)

    def run():
        norm_rope.fused_q_norm_rope(source, first, 1e-6, freqs, positions)
        norm_rope.fused_q_norm_rope(first, second, 1e-6, freqs, positions)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        source.normal_()
        positions.random_(0, 128)
        graph.replay()
        expected = source.clone()
        for _ in range(2):
            expected = (
                expected.float() * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + 1e-6)
            ).bfloat16()
            expected[..., -64:] = rope_reference(expected, freqs, positions).bfloat16()
        torch.testing.assert_close(second, expected, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("topk", [128, 256, 1024])
@pytest.mark.parametrize("width", [64, 640, 4096])
def test_topk_uses_configured_output_capacity(topk, width):
    scores = torch.randn(2, width, device="cuda")
    lengths = torch.tensor([width // 2, width], dtype=torch.int32, device="cuda")
    pages = torch.randint(0, 1000, (2, (width + 63) // 64), dtype=torch.int32, device="cuda")
    out = torch.empty((2, topk), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(out)
    topk_transform.topk_transform(scores, lengths, pages, out, 64, raw)
    assert_topk(scores, lengths, pages, out, raw, 64)
