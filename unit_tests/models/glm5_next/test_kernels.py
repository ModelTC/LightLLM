import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch
import triton

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import set_env_start_args
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda import (
    chunk_kda_with_fused_gate,
    fused_kda_gate_chunk_cumsum,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode import fused_recurrent_kda
from lightllm.models.glm5_next.triton_kernel.kpool import (
    compress_pools,
    gather_pools,
    gather_paged_pools,
    get_pool_ranges,
    expand_topk,
)
from lightllm.models.glm5_next.triton_kernel.index_quant import hadamard_transform_quant_fp8


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def setup():
    torch.manual_seed(1525)
    set_env_start_args(dataclasses.asdict(StartArgs()))
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))


def _reference_kda(q, k, v, gate, beta, a, bias, state):
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6) / q.shape[-1] ** 0.5
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    decay = (-5 * torch.sigmoid(a.exp()[:, None] * (gate.float() + bias))).exp()
    state = state * decay[..., None]
    delta = (v.float() - torch.einsum("hkv,hk->hv", state, k)) * beta.float().sigmoid()[:, None]
    state = state + k[..., None] * delta[:, None, :]
    return torch.einsum("hkv,hk->hv", state, q), state


@pytest.mark.parametrize("tokens", [1, 3, 65, 129])
def test_kda_chunk_and_decode_match_recurrence(tokens):
    heads, dim = 2, 128
    rand = lambda *shape: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    q, k, v, gate = [rand(1, tokens, heads, dim) for _ in range(4)]
    beta = rand(1, tokens, heads)
    a = torch.randn(heads, device="cuda")
    bias = torch.randn(heads, dim, device="cuda")
    initial = torch.randn(1, heads, dim, dim, device="cuda") * 0.1
    expected = []
    state = initial[0].clone()
    for i in range(tokens):
        out, state = _reference_kda(q[0, i], k[0, i], v[0, i], gate[0, i], beta[0, i], a, bias, state)
        expected.append(out)
    expected = torch.stack(expected).unsqueeze(0)
    actual, final = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=gate,
        beta=beta.float().sigmoid(),
        A_log=a,
        g_bias=bias.flatten(),
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        safe_gate=True,
    )
    torch.testing.assert_close(actual.float(), expected, atol=4e-3, rtol=3e-2)
    torch.testing.assert_close(final[0], state, atol=8e-3, rtol=3e-2)
    # Reuse a nonzero request slot; the neighboring requests must stay intact.
    states = torch.randn(4, heads, dim, dim, device="cuda")
    states[2] = initial[0]
    unchanged = states[[0, 1, 3]].clone()
    for i in range(tokens):
        out, _ = fused_recurrent_kda(
            q[:, i : i + 1],
            k[:, i : i + 1],
            v[:, i : i + 1],
            gate[:, i : i + 1].reshape(1, 1, -1),
            beta[:, i : i + 1],
            a,
            bias.flatten(),
            states,
            torch.tensor([2], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(out[0, 0].float(), expected[0, i], atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(states[2], state, atol=2e-5, rtol=2e-4)
    assert torch.equal(states[[0, 1, 3]], unchanged)


@pytest.mark.parametrize("seq_lens", [(65,), (3, 65, 129)])
@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_kda_gate_cumsum_packed_shape(seq_lens, safe_gate, strided):
    heads, dim, chunk_size = 2, 128, 64
    if strided:
        # All three input strides differ from contiguous [T, H, D]; empty_like
        # allocates a contiguous output for this sliced, non-dense input view.
        storage = torch.randn(sum(seq_lens) * 2, heads * 2, dim * 2, device="cuda", dtype=torch.bfloat16)
        raw_g = storage[::2, ::2, ::2]
    else:
        raw_g = torch.randn(sum(seq_lens), heads, dim, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda")
    bias = torch.randn(heads, dim, device="cuda") if len(seq_lens) > 1 else None
    cu_seqlens = torch.tensor([0, *seq_lens], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    lower_bound = -3.0

    actual = fused_kda_gate_chunk_cumsum(
        raw_g,
        A_log=a_log,
        g_bias=bias.flatten() if bias is not None else None,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    gate_input = raw_g.float() + (bias if bias is not None else 0)
    amplitude = a_log.exp()[None, :, None]
    if safe_gate:
        log_gate = lower_bound * torch.sigmoid(amplitude * gate_input)
    else:
        log_gate = -amplitude * torch.nn.functional.softplus(gate_input)
    expected = torch.empty_like(log_gate)
    seq_start = 0
    for seq_len in seq_lens:
        for offset in range(0, seq_len, chunk_size):
            start = seq_start + offset
            end = seq_start + min(offset + chunk_size, seq_len)
            expected[start:end] = log_gate[start:end].cumsum(0) / math.log(2)
        seq_start += seq_len

    assert actual.shape == raw_g.shape
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "q_lens",
    [(3, 2, 4), (0, 1, 0, 1, 0), (1,) * 64, (257, 1, 513, 0), (8129,) + (1,) * 63],
)
def test_kpool_ranges_cuda_graph_with_changing_query_boundaries(q_lens):
    # Move the longest query across the batch while replaying the same graph.
    # Empty queries must not use the one-query-per-request decode shortcut.
    tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    lengths = torch.empty(sum(q_lens), device="cuda", dtype=torch.int32)
    cu_q_lens = torch.empty(len(q_lens) + 1, device="cuda", dtype=torch.int32)
    max_pools = 4096

    def set_inputs(counts, phase):
        cu, visible_lengths, pool_starts = [0], [], []
        for batch, count in enumerate(counts):
            prefix = batch * 7 + phase
            visible_lengths.extend(range(prefix + 1, prefix + count + 1))
            pool_starts.extend([batch * max_pools] * count)
            cu.append(cu[-1] + count)
        lengths.copy_(tensor(visible_lengths))
        cu_q_lens.copy_(tensor(cu))
        expected_lengths = tensor([length // 4 for length in visible_lengths])
        expected_starts = tensor(pool_starts)
        return expected_starts, expected_starts + expected_lengths, expected_lengths

    expected = set_inputs(q_lens, 0)
    actual = get_pool_ranges(lengths, cu_q_lens, max(q_lens), max_pools)
    for output, reference in zip(actual, expected):
        assert torch.equal(output, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = get_pool_ranges(lengths, cu_q_lens, max(q_lens), max_pools)
    expected = set_inputs(q_lens[::-1], 3)
    graph.replay()
    for output, reference in zip(actual, expected):
        assert torch.equal(output, reference)


@pytest.mark.parametrize("cuda_graph", [False, True])
def test_kpool_chunk_boundaries_and_fragmented_token_kv(cuda_graph):
    # Unaligned chunks, reordered requests, and pools spanning old/new tails.
    seqs = [9, 7]
    ragged = torch.randperm(40, device="cuda", dtype=torch.int32)[: sum(seqs)]
    req_idx = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    table = torch.zeros(5, 12, device="cuda", dtype=torch.int32)
    table[1, :9], table[3, :7] = ragged[:9], ragged[9:]
    source = torch.randn(sum(seqs), 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda")
    packed_storage = torch.zeros(40, 1, 584, device="cuda", dtype=torch.bfloat16)
    packed = packed_storage.view(torch.uint8)[:, :, -132:]
    tail = torch.randn(5, 4, 256, device="cuda", dtype=torch.bfloat16)
    unchanged = tail[[0, 2, 4]].clone()
    tensor = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
    for chunks in [[(1, 0, 3), (3, 0, 2)], [(3, 2, 3), (1, 3, 9)], [(3, 3, 7)]]:
        raw, lengths, starts, locations = [], [], [], []
        cu_q_lens = [0]
        for req, first, end in chunks:
            offset = 0 if req == 1 else 9
            raw.append(source[offset + first : offset + end])
            lengths.extend(range(first + 1, end + 1))
            starts.extend([len(locations)] * (end - first))
            locations.extend(table[req, :end].tolist())
            cu_q_lens.append(cu_q_lens[-1] + end - first)
        args = (
            torch.cat(raw),
            tail,
            packed,
            ape,
            tensor(lengths),
            tensor(starts),
            tensor(locations),
            tensor([r for r, _, _ in chunks]),
            tensor(cu_q_lens),
            tensor([end for _, _, end in chunks]),
        )
        max_q_len = max(end - first for _, first, end in chunks)
        if cuda_graph:
            # Compile before capture, then restore this chunk's input cache state.
            saved_tail, saved_packed = tail.clone(), packed.clone()
            compress_pools(*args, max_q_len=max_q_len)
            tail.copy_(saved_tail)
            packed.copy_(saved_packed)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compress_pools(*args, max_q_len=max_q_len)
            graph.replay()
        else:
            compress_pools(*args, max_q_len=max_q_len)
        for req, _, end in chunks:
            offset = 0 if req == 1 else 9
            assert torch.equal(tail[req, : end % 4], source[offset + end // 4 * 4 : offset + end])
        assert torch.equal(tail[[0, 2, 4]], unchanged)
        # Model the full KV copy performed by cache offload/load or request move.
        packed_storage = packed_storage.clone()
        packed = packed_storage.view(torch.uint8)[:, :, -132:]
    keys, scales = gather_pools(packed, table, req_idx, torch.tensor(seqs, device="cuda", dtype=torch.int32), 3)
    for batch, start in enumerate([0, 9]):
        for group in range(seqs[batch] // 4):
            values = source[start + group * 4 : start + group * 4 + 4]
            expected = (values[:, :128].float() * (values[:, 128:].float() + ape).softmax(0)).sum(0).bfloat16()
            expected_key, expected_scale = hadamard_transform_quant_fp8(expected[None], scale=128 ** -0.5)
            torch.testing.assert_close(keys[batch * 3 + group].float(), expected_key[0].float(), atol=0, rtol=0)
            torch.testing.assert_close(scales[batch * 3 + group], expected_scale[0, 0], atol=0, rtol=0)
    lengths = torch.tensor(seqs, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 9], device="cuda", dtype=torch.int32)
    groups = torch.tensor([[1, 0], [0, -1]], device="cuda", dtype=torch.int32)
    out, relative = expand_topk(groups, lengths, starts, ragged, topk=8)
    for batch, expected in enumerate([[4, 5, 6, 7, 0, 1, 2, 3, 8], list(range(7))]):
        assert relative[batch, : len(expected)].tolist() == expected
        assert (
            out[batch, : len(expected)].tolist()
            == ragged[starts[batch].item() + torch.tensor(expected, device="cuda")].tolist()
        )
        assert (out[batch, len(expected) :] == -1).all()


def test_kpool_decode_cuda_graph_and_padding():
    # Replay one graph across all four tail phases, with reordered requests
    # and repeated padding slots. Padding must not race on the hold state.
    source = torch.randn(2, 8, 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda")
    tail = torch.zeros(5, 4, 256, device="cuda", dtype=torch.bfloat16)
    storage = torch.zeros(32, 1, 584, device="cuda", dtype=torch.bfloat16)
    packed = storage.view(torch.uint8)[:, :, -132:]
    table = torch.randperm(32, device="cuda", dtype=torch.int32).view(4, 8)
    raw = torch.zeros(4, 256, device="cuda", dtype=torch.bfloat16)
    req_idx = torch.tensor([1, 3, 4, 4], device="cuda", dtype=torch.int32)
    lengths = torch.ones(4, device="cuda", dtype=torch.int32)
    starts = torch.arange(4, device="cuda", dtype=torch.int32) * 8
    ragged = table.flatten().clone()
    cu_q_lens = torch.arange(5, device="cuda", dtype=torch.int32)

    def run():
        compress_pools(raw, tail, packed, ape, lengths, starts, ragged, req_idx, cu_q_lens, lengths, max_q_len=1)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    tail.zero_()
    for pos in range(8):
        order = [pos % 2, 1 - pos % 2]
        req_idx[:2] = torch.tensor([1 if i == 0 else 3 for i in order], device="cuda")
        raw[:2] = source[order, pos]
        lengths.fill_(pos + 1)
        ragged[:16] = table[order].flatten()
        graph.replay()
        for batch, req in enumerate([1, 3]):
            pending = (pos + 1) % 4
            assert torch.equal(tail[req, :pending], source[batch, (pos + 1) // 4 * 4 : pos + 1])
            if pending == 0:
                values = source[batch, pos - 3 : pos + 1]
                pooled = (values[:, :128].float() * (values[:, 128:].float() + ape).softmax(0)).sum(0).bfloat16()
                key, scale = hadamard_transform_quant_fp8(pooled[None], 128 ** -0.5)
                actual = packed[table[batch, pos].long(), 0]
                assert torch.equal(actual[:128], key.view(torch.uint8)[0])
                assert torch.equal(actual[128:].view(torch.float32), scale.flatten())
        assert not tail[[0, 2, 4]].any()


@pytest.mark.parametrize("max_pools", [640, 262144])
def test_gather_paged_pools_valid_pages_and_graph_replay(max_pools):
    storage = torch.zeros(64, 1, 584, device="cuda", dtype=torch.bfloat16)
    packed = storage.view(torch.uint8)[:, :, -132:]
    source_keys = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    source_scales = torch.rand(64, device="cuda") + 0.1
    packed[:, 0, :128] = source_keys
    packed[:, 0, 128:] = source_scales.view(torch.uint8).view(64, 4)
    table = torch.full((4, max_pools * 4), -1, device="cuda", dtype=torch.int32)
    locations = torch.randint(0, 64, (4, max_pools), device="cuda", dtype=torch.int32)
    table[:, 3::4] = locations
    req_idx = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([max_pools, 3, 0], device="cuda", dtype=torch.int32)

    def check(pages, block_table):
        page_bytes = pages.view(pages.shape[0], -1)
        for row, (req, length) in enumerate(zip(req_idx.tolist(), lengths.tolist())):
            page_ids = block_table[row, : triton.cdiv(length, 64)].long()
            keys = page_bytes[page_ids, : 64 * 128].reshape(-1, 128)
            scales = page_bytes[page_ids, 64 * 128 :].contiguous().view(torch.float32).flatten()
            locs = locations[req, :length].long()
            assert torch.equal(keys[:length], source_keys[locs])
            assert torch.equal(scales[:length], source_scales[locs])
            assert not keys[length:].any()
            assert (scales[length:] == 1).all()

    check(*gather_paged_pools(packed, table, req_idx, lengths, max_pools))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pages, block_table = gather_paged_pools(packed, table, req_idx, lengths, max_pools)
    # Empty HOLD rows must do no page writes, even with a 1M graph capacity.
    pages.fill_(127)
    lengths.zero_()
    graph.replay()
    assert (pages == 127).all()
    for counts in ([513, 65, max_pools], [1, 0, 3]):
        lengths.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        req_idx.copy_(torch.tensor([1, 3, 0], device="cuda", dtype=torch.int32))
        graph.replay()
        check(pages, block_table)


@pytest.mark.parametrize("max_pools", [65535, 65536, 262144])
def test_gather_pools_long_context_cuda_graph(max_pools):
    # 1M-token graph warmup pads to 262144 pools even for empty hold requests.
    # Also gather real pools at the end of that range from fragmented token KV.
    storage = torch.zeros(64, 1, 584, device="cuda", dtype=torch.bfloat16)
    packed = storage.view(torch.uint8)[:, :, -132:]
    source_keys = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    source_scales = torch.rand(64, device="cuda") + 0.1
    packed[:, 0, :128] = source_keys
    packed[:, 0, 128:] = source_scales.view(torch.uint8).view(64, 4)
    table = torch.full((4, max_pools * 4), -1, device="cuda", dtype=torch.int32)
    locations = torch.randint(0, 64, (4, max_pools), device="cuda", dtype=torch.int32)
    table[:, 3::4] = locations
    req_idx = torch.tensor([2, 0, 3], device="cuda", dtype=torch.int32)
    seq_len = torch.tensor([max_pools * 4, 14, 2], device="cuda", dtype=torch.int32)

    def check(keys, scales):
        keys = keys.view(torch.uint8).view(3, max_pools, 128)
        scales = scales.view(3, max_pools)
        for batch, (req, count) in enumerate([(2, max_pools), (0, 3), (3, 0)]):
            locs = locations[req, :count].long()
            assert torch.equal(keys[batch, :count], source_keys[locs])
            assert torch.equal(scales[batch, :count], source_scales[locs])
            assert not keys[batch, count:].any()
            assert (scales[batch, count:] == 1).all()

    check(*gather_pools(packed, table, req_idx, seq_len, max_pools))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keys, scales = gather_pools(packed, table, req_idx, seq_len, max_pools)
    # Replay with the same short hold lengths used by model startup.
    original_lengths = seq_len.clone()
    seq_len.fill_(2)
    graph.replay()
    assert not keys.view(torch.uint8).any()
    assert (scales == 1).all()
    seq_len.copy_(original_lengths)
    graph.replay()
    check(keys, scales)


def test_kpool_packed_kv_above_two_gib():
    # Automatic KV sizing on H200 can put pool keys beyond a 32-bit byte offset.
    loc = 2 ** 31 // (584 * 2) + 1
    storage = torch.empty(loc + 1, 1, 584, device="cuda", dtype=torch.bfloat16)
    packed = storage.view(torch.uint8)[:, :, -132:]
    raw = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    tail = torch.zeros(2, 4, 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda")
    lengths = torch.arange(1, 5, device="cuda", dtype=torch.int32)
    zero = torch.zeros(4, device="cuda", dtype=torch.int32)
    ragged = torch.tensor([0, 1, 2, loc], device="cuda", dtype=torch.int32)
    cu_q_lens = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    compress_pools(raw, tail, packed, ape, lengths, zero, ragged, zero[:1], cu_q_lens, lengths[-1:], max_q_len=4)
    keys, scales = gather_pools(packed, ragged[None], zero[:1], lengths[-1:], 1)
    pooled = (raw[:, :128].float() * (raw[:, 128:].float() + ape).softmax(0)).sum(0).bfloat16()
    expected_key, expected_scale = hadamard_transform_quant_fp8(pooled[None], scale=128 ** -0.5)
    assert torch.equal(keys.view(torch.uint8), expected_key.view(torch.uint8))
    assert torch.equal(scales, expected_scale.flatten())


@pytest.mark.parametrize("tp_world_size", [1, 4])
def test_mhc_keeps_streams_through_decode_autotuning(monkeypatch, tp_world_size):
    from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel

    hidden = 4096
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", str(tp_world_size))
    pre_infer = Glm5NextTpPartModel.pre_layer_infer_class({"hc_mult": 4})
    embeddings = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
    input_ids = torch.zeros(1, device="cuda", dtype=torch.long)
    pre_weight = SimpleNamespace(wte_weight_=lambda input_ids, alloc_func: embeddings[input_ids])
    pre_weight.wte_weight_.weight = embeddings
    pre_weight.wte_weight_.tp_vocab_start_id = 0
    pre_weight.wte_weight_.tp_vocab_end_id = 1
    infer_state = SimpleNamespace(dist_group=None, multimodal_params=[])

    def all_reduce(input_embeddings, **kwargs):
        # TP communication must operate on the original embedding width.
        assert input_embeddings.shape == (1, hidden)
        input_embeddings.mul_(tp_world_size)

    monkeypatch.setattr("lightllm.models.llama.layer_infer.pre_layer_infer.all_reduce", all_reduce)
    monkeypatch.setattr("lightllm.models.qwen_vl.layer_infer.pre_layer_infer.all_reduce", all_reduce)
    expected_streams = (embeddings * tp_world_size).unsqueeze(1).expand(-1, 4, -1)
    weight = SimpleNamespace(
        att_norm_weight_=SimpleNamespace(weight=torch.ones(hidden, device="cuda", dtype=torch.bfloat16)),
        ffn_norm_weight_=SimpleNamespace(weight=torch.ones(hidden, device="cuda", dtype=torch.bfloat16)),
    )
    for prefix in ("attn", "ffn"):
        setattr(weight, f"hc_{prefix}_fn", SimpleNamespace(weight=torch.randn(24, 4 * hidden, device="cuda") * 0.005))
        setattr(weight, f"hc_{prefix}_base", SimpleNamespace(weight=torch.zeros(24, device="cuda")))
        setattr(weight, f"hc_{prefix}_scale", SimpleNamespace(weight=torch.ones(3, device="cuda")))
    layer = object.__new__(Glm5NextTransformerLayerInfer)
    layer.use_mhc = True
    layer.embed_dim_, layer.mhc_streams = hidden, 4
    layer.num_hidden_layers, layer.autotune_layer_num = 5, 4
    layer.eps_, layer.hc_eps, layer.hc_sinkhorn_iters = 1e-5, 1e-6, 20
    layer.token_attention_forward = layer.context_attention_forward = layer._ffn = lambda x, *_: x * 0.1
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        x = pre_infer.token_forward(input_ids, infer_state, pre_weight)
        torch.testing.assert_close(x.view(1, 4, hidden), expected_streams, atol=0, rtol=0)
        for i in range(5):
            layer.layer_num_ = i
            x = layer.token_forward(x, None, weight)
            assert x.shape == (1, hidden if i == 4 else 4 * hidden)
    with Autotuner.autotune_warmup():
        x = pre_infer.context_forward(input_ids, infer_state, pre_weight)
        torch.testing.assert_close(x.view(1, 4, hidden), expected_streams, atol=0, rtol=0)
        for i in range(4):
            layer.layer_num_ = i
            x = layer.context_forward(x, None, weight)
            assert x.shape == (1, hidden if i == 3 else 4 * hidden)


def test_shared_chunk_kernel_preserves_gdn_natural_log_decay():
    from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import chunk_gated_delta_rule

    tokens, heads, dim = 67, 2, 128
    q, k, v = [torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q = torch.nn.functional.normalize(q.float(), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(k.float(), dim=-1).bfloat16()
    gate = -torch.rand(1, tokens, heads, device="cuda")
    beta = torch.rand_like(gate)
    state = torch.randn(1, heads, dim, dim, device="cuda") * 0.1
    initial = state.clone()
    expected = []
    for i in range(tokens):
        state *= gate[:, i, :, None, None].exp()
        delta = (v[:, i].float() - torch.einsum("bhkv,bhk->bhv", state, k[:, i].float())) * beta[:, i, :, None]
        state += k[:, i, :, :, None].float() * delta[:, :, None, :]
        expected.append(torch.einsum("bhkv,bhk->bhv", state, q[:, i].float()) / dim ** 0.5)
    out, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(out.float(), torch.stack(expected, 1), atol=4e-3, rtol=3e-2)
    torch.testing.assert_close(final, state, atol=8e-3, rtol=3e-2)


@pytest.mark.parametrize("heads", [16, 64, 128])
def test_nope_attention_native_512_and_cuda_graph(heads):
    from lightllm.common.basemodel.attention.base_att import AttControl
    from lightllm.common.basemodel.attention.nsa.glm5_next import Glm5NextSparsePrefillState, Glm5NextSparseDecodeState

    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 NoPE requires Hopper")
    # The 72-element tail stores indexer data, not a zero RoPE embedding.
    packed = torch.randn(4096, 1, 584, dtype=torch.bfloat16, device="cuda")
    kv = packed[:, :, :512]
    # Match the head-major projection's non-contiguous query layout.
    q = torch.randn(heads, 9, 512, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    lengths = torch.tensor([0, 1, 3, 127, 128, 511, 2048, 2049, 2051], dtype=torch.int32, device="cuda")
    indexes = torch.randint(0, 4096, (9, 2176), dtype=torch.int32, device="cuda")
    offsets = torch.arange(indexes.shape[1], device="cuda")
    indexes.masked_fill_(offsets[None, :] >= lengths[:, None], -1)

    def reference():
        expected = []
        for i, length in enumerate(lengths.tolist()):
            keys = kv[indexes[i, :length].long(), 0].float()
            expected.append((q[i].float() @ keys.T * 0.0625).softmax(-1) @ keys)
        return torch.stack(expected)

    expected = reference()
    control = AttControl(
        nsa_prefill_dict={"topk_mem_indices": indexes, "softmax_scale": 0.0625, "kv_lora_rank": kv.shape[-1]}
    )
    prefill_state = Glm5NextSparsePrefillState()
    prefill = prefill_state._nsa_prefill_att(q, kv, control)
    torch.testing.assert_close(prefill.float(), expected, atol=0.012, rtol=0.015)
    decode = Glm5NextSparseDecodeState(
        infer_state=SimpleNamespace(
            b1_cu_q_seq_len=torch.arange(10, dtype=torch.int32, device="cuda"), max_q_seq_len=1
        ),
        nsa_cache_seqlens=lengths,
        nsa_cu_seqlens_k_new=torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)),
    )
    control.nsa_decode_dict = control.nsa_prefill_dict
    out = decode._nsa_decode_att((q, q[..., :0]), kv, control)
    torch.testing.assert_close(out.float(), expected, atol=0.012, rtol=0.015)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prefill_state._nsa_prefill_att(q, kv, control)
        decode._nsa_decode_att((q, q[..., :0]), kv, control)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_prefill = prefill_state._nsa_prefill_att(q, kv, control)
        graph_decode = decode._nsa_decode_att((q, q[..., :0]), kv, control)
    graph.replay()
    torch.testing.assert_close(graph_prefill.float(), expected, atol=0.012, rtol=0.015)
    torch.testing.assert_close(graph_decode.float(), expected, atol=0.012, rtol=0.015)

    q.normal_()
    packed.normal_()
    lengths.copy_(lengths.roll(1))
    decode.nsa_cu_seqlens_k_new[1:].copy_(lengths.cumsum(0, dtype=torch.int32))
    indexes.random_(0, 4096)
    indexes.masked_fill_(offsets[None, :] >= lengths[:, None], -1)
    graph.replay()
    expected = reference()
    torch.testing.assert_close(graph_prefill.float(), expected, atol=0.012, rtol=0.015)
    torch.testing.assert_close(graph_decode.float(), expected, atol=0.012, rtol=0.015)


@pytest.mark.parametrize("max_kv_seq_len", [2065, 1 << 20])
def test_kpool_indexer_long_prefill_and_cached_decode(max_kv_seq_len):
    from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer

    tokens, heads, dim = 2065, 32, 128
    hidden = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn(dim, heads * dim, device="cuda", dtype=torch.bfloat16) * 0.1
    ape = torch.randn(4, dim, device="cuda")
    weights = SimpleNamespace(
        wk_proj_=SimpleNamespace(mm=lambda x: x),
        k_norm_=lambda x, eps: x,
        index_kpool_compress_gate=SimpleNamespace(mm=lambda x: x * 0.25),
        index_kpool_compress_ape=SimpleNamespace(weight=ape),
        wq_b_proj_=SimpleNamespace(mm=lambda x: x @ q_weight),
        weights_proj_=SimpleNamespace(mm=lambda x: torch.ones(x.shape[0], heads, device=x.device)),
    )
    storage = torch.zeros(tokens + 9, 1, 584, device="cuda", dtype=torch.bfloat16)
    tail = torch.zeros(2, 4, 256, device="cuda", dtype=torch.bfloat16)
    ragged = torch.randperm(tokens + 9, device="cuda", dtype=torch.int32)[:tokens]
    manager = SimpleNamespace(
        get_indexer_k_buffer=lambda _: storage.view(torch.uint8)[:, :, -132:],
    )
    infer = SimpleNamespace(
        mem_manager=manager,
        is_prefill=True,
        b_mtp_index=torch.zeros(1, device="cuda", dtype=torch.int32),
        mem_index=ragged,
        # A large capacity also exercises query chunking and uninitialized
        # logits beyond each query's actual pool range.
        max_kv_seq_len=max_kv_seq_len,
        req_manager=SimpleNamespace(req_to_token_indexs=ragged[None], get_indexer_tail_buffer=lambda _: tail),
        b_req_idx=torch.zeros(1, device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([tokens], device="cuda", dtype=torch.int32),
        b1_cu_q_seq_len=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
        max_q_seq_len=tokens,
    )
    state = SimpleNamespace(
        lengths=torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32),
        ks=torch.zeros(tokens, device="cuda", dtype=torch.int32),
        ragged_mem_index=ragged,
    )
    indexer = Glm5NextNsaInfer(
        0, {"index_topk": 2048, "index_n_heads": heads, "index_head_dim": dim, "rms_norm_eps": 1e-5}, 1
    )
    _, full = indexer._get_indices(hidden, hidden, infer, state, weights)
    assert full[-1, 2048].item() == tokens - 1  # Always-selected incomplete tail.
    assert full[0, 0].item() == 0 and (full[0, 1:] == -1).all()
    assert full[-1, :2049].unique().numel() == 2049
    assert (full[-1, :2049] < tokens).all()
    # Restore an aligned prefix using only token KV, then decode across the
    # next pool boundary. The runtime tail starts empty, as on a page restore.
    prefix = 2060
    storage = storage.clone()
    storage[ragged[prefix:].long()] = 0
    tail.zero_()
    infer.is_prefill = False
    infer.b1_cu_q_seq_len[1] = 1
    infer.max_q_seq_len = 1
    for pos in range(prefix, tokens):
        infer.mem_index = ragged[pos : pos + 1]
        infer.max_kv_seq_len = pos + 1
        infer.b_seq_len.fill_(pos + 1)
        state.lengths = torch.tensor([pos + 1], device="cuda", dtype=torch.int32)
        state.ks = torch.zeros_like(state.lengths)
        _, decoded = indexer._get_indices(hidden[pos : pos + 1], hidden[pos : pos + 1], infer, state, weights)
        valid = 2048 + (pos + 1) % 4
        assert set(decoded[0, :valid].tolist()) == set(full[pos, :valid].tolist())

    pool_count = tokens // 4
    pool_values = hidden[: pool_count * 4].view(pool_count, 4, dim)
    pooled = (pool_values.float() * (pool_values.float() * 0.25 + ape).softmax(1)).sum(1).bfloat16()
    k_fp8, k_scale = hadamard_transform_quant_fp8(pooled, dim ** -0.5)
    query = (hidden[-1:] @ q_weight).view(heads, dim)
    q_fp8, q_scale = hadamard_transform_quant_fp8(query, dim ** -0.5)
    logits = (q_fp8.float() @ k_fp8.float().T * k_scale.flatten()).clamp_min(0)
    scores = (logits * q_scale * (heads ** -0.5 * dim ** -0.5)).sum(0)
    expected_groups = set(scores.topk(512).indices.tolist())
    assert set((decoded[0, :2048:4] // 4).tolist()) == expected_groups
