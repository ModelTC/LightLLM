import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from unit_tests.common.attention.backends.step_fixture import make_infer_state, make_resources
from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend, FA3GQAPrefillBackend
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from unit_tests.common.attention.backends.groups_runner import GroupsRunner


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or flash_attn_with_kvcache is None, reason="FA3 (sgl_kernel) is required"
)
PAD = 7  # padding request id; its token-map row is valid but its output is ignored
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# The FP8 references use the same dequantized K/V and FP8-rounded q as the kernel, but FA3's FP8 path also
# rounds the softmax probabilities to E4M3 (relative error <= 2**-4) before P @ V, which moves each output by
# up to ~6% of the probability-weighted |V| (observed: <= 0.055 where |expected| reaches 2.3). The V scales
# are ~2x the K scales, so a swapped K/V descale, like a lost q descale, moves outputs by far more.
FP8_TOL = 0.06


def reference(q, k, v, token_map, rows, *, causal=True, window=(-1, -1), sinks=None):
    """rows: per query row (request id, keys the row may see causally, keys of its request)."""
    out = []
    for i, (req_id, own_len, request_len) in enumerate(rows):
        slots = token_map[req_id, :request_len].long()
        keys = k[slots].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        values = v[slots].float().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        scores = torch.einsum("hd,khd->hk", q[i].float(), keys) * q.shape[-1] ** -0.5
        pos = own_len - 1
        key_pos = torch.arange(request_len, device=q.device)
        mask = key_pos <= pos if causal else torch.ones_like(key_pos, dtype=torch.bool)
        if window != (-1, -1):
            mask &= (key_pos >= pos - window[0]) & (key_pos <= pos + window[1])
        scores = scores.masked_fill(~mask[None, :], float("-inf"))
        if sinks is not None:
            scores = torch.cat((scores, sinks.float()[:, None]), dim=1)
            values = torch.cat((values, values.new_zeros(1, *values.shape[1:])), dim=0)
        out.append(torch.einsum("hk,khd->hd", scores.softmax(-1), values))
    return torch.stack(out)


def paged_token_map(requests: int, page_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pages = torch.randperm(requests * 512 // page_size, device="cuda", dtype=torch.int32, generator=generator)
    pages = pages.view(requests, -1)
    return (pages[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(requests, 512)


def make_batch(token_map, req_ids, lengths, query_lengths=None, **kwargs):
    query_lengths = query_lengths or [1] * len(lengths)
    cu_q = torch.tensor([0, *query_lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_kv = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return make_infer_state(
        req_to_token_indexs=token_map,
        b_req_idx=torch.tensor(req_ids, device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        b1_cu_q_seq_len=cu_q,
        b1_cu_kv_seq_len=cu_kv,
        max_q_seq_len=max(query_lengths),
        pad_req_idx=PAD,
        **kwargs,
    )


def make_cache(kv_heads, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(8 * 512, 2 * kv_heads, 64, device="cuda", dtype=dtype, generator=generator)
    return kv, kv[:, :kv_heads], kv[:, kv_heads:]


def make_fp8_cache(kv_heads, seed):
    """Return per-head FP8 K/V cache bytes, their dequantized values and the float32 [kv_heads] scales."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(8 * 512, 2, kv_heads, 64, device="cuda", generator=generator)
    # Per-head scales differ between heads and between K and V; |kv| / scale stays inside E4M3's 448.
    scales = torch.stack(
        (torch.linspace(0.02, 0.025, kv_heads, device="cuda"), torch.linspace(0.04, 0.045, kv_heads, device="cuda"))
    )
    encoded = (kv / scales[None, :, :, None]).to(torch.float8_e4m3fn)
    decoded = encoded.float() * scales[None, :, :, None]
    k_scale, v_scale = scales.unbind(0)
    return (
        encoded[:, 0].view(torch.uint8),
        encoded[:, 1].view(torch.uint8),
        decoded[:, 0],
        decoded[:, 1],
        k_scale,
        v_scale,
    )


def fp8_round_q(q, kv_heads, groups, static_scale=None):
    """Round q to FP8 as FA3's per-KV-head quantization does: the static [kv_heads] scale, or one dynamic
    amax / FP8_MAX scale per KV head and query group."""
    shape = q.shape
    q = q.float().view(shape[0], kv_heads, -1)
    out = torch.empty_like(q)
    for rows in groups:
        if static_scale is None:
            amax = q[rows].abs().amax(dim=(0, 2))
            scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        else:
            scale = static_scale.float()
        encoded = (q[rows] / scale[None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        out[rows] = encoded.float() * scale[None, :, None]
    return out.view(shape)


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize(
    "causal,window,sinks", [(True, (-1, -1), False), (True, (15, 0), True), (False, (6, 6), False)]
)
@pytest.mark.parametrize("cached", [False, True])
def test_prefill(page_size, causal, window, sinks, cached):
    dtype, q_heads, kv_heads = torch.bfloat16, 8, 2
    lengths, ready, req_ids = [33, 9, 259], ([16, 5, 128] if cached else [0, 0, 0]), [2, 0, 1]
    token_map = paged_token_map(8, page_size, seed=0)
    kv, k, v = make_cache(kv_heads, dtype, seed=1)
    before = kv.clone()
    query_lengths = [n - r for n, r in zip(lengths, ready)]
    q = torch.randn(sum(query_lengths), q_heads, 64, device="cuda", dtype=dtype)
    sink_logits = torch.randn(q_heads, device="cuda", dtype=dtype) if sinks else None
    layer_config = GQALayerConfig(
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=64,
        dtype=dtype,
        sliding_window=window,
        causal=causal,
        has_sinks=sinks,
    )
    batch = make_batch(token_map, req_ids, lengths, query_lengths, b_ready_cache_len=torch.tensor(ready).cuda())
    backend = FA3GQAPrefillBackend(layer_config)
    att_state = backend.create_att_state(batch, make_resources(page_size=page_size))
    out = torch.empty_like(q)
    assert backend.forward(att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v, sinks=sink_logits), out=out) is out
    rows = [(r, pos + 1, n) for r, n, c in zip(req_ids, lengths, ready) for pos in range(c, n)]
    expected = reference(q, k, v, token_map, rows, causal=causal, window=window, sinks=sink_logits)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(kv, before, atol=0, rtol=0)


@pytest.mark.parametrize("static_q", [False, True])
def test_fp8_prefill(static_q):
    q_heads, kv_heads = 8, 2
    lengths, ready, req_ids = [33, 9, 259], [16, 5, 128], [2, 0, 1]
    token_map = paged_token_map(8, 1, seed=0)
    k, v, k_ref, v_ref, k_scale, v_scale = make_fp8_cache(kv_heads, seed=1)
    query_lengths = [n - r for n, r in zip(lengths, ready)]
    generator = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(sum(query_lengths), q_heads, 64, device="cuda", generator=generator).bfloat16()
    q_scale = q.float().abs().view(q.shape[0], kv_heads, -1).amax(dim=(0, 2)) / FP8_MAX if static_q else None
    layer_config = GQALayerConfig(
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=64,
        dtype=torch.bfloat16,
        kv_quant="fp8_per_head",
        has_q_scale=static_q,
    )
    batch = make_batch(token_map, req_ids, lengths, query_lengths, b_ready_cache_len=torch.tensor(ready).cuda())
    backend = FA3GQAPrefillBackend(layer_config)
    att_state = backend.create_att_state(batch, make_resources(page_size=16))
    assert att_state.page_size == 1, "an FP8 cache runs with single-token pages"
    out = torch.empty_like(q)
    layer_inputs = GQALayerInputs(
        q=q, k_cache=k, v_cache=v, kv_scales=GQAKVScales(k_scale=k_scale, v_scale=v_scale, q_scale=q_scale)
    )
    assert backend.forward(att_state, layer_inputs, out=out) is out
    # Dynamic q scales are per prefill request.
    starts = [0, *torch.tensor(query_lengths).cumsum(0).tolist()]
    q_ref = fp8_round_q(q, kv_heads, [list(range(s, e)) for s, e in zip(starts, starts[1:])], q_scale)
    rows = [(r, pos + 1, n) for r, n, c in zip(req_ids, lengths, ready) for pos in range(c, n)]
    expected = reference(q_ref, k_ref, v_ref, token_map, rows)
    torch.testing.assert_close(out.float(), expected, atol=FP8_TOL, rtol=FP8_TOL)


# Decode layouts: (b_req_idx per row, b_seq_len per row, rows per request, dynamic rows).
LAYOUTS = {
    "normal": ([2, 0, 1], [34, 10, 260], 1, False),
    # Two rows per request; the last row of each request sees its longest prefix.
    "fixed": ([2, 2, 0, 0], [33, 34, 9, 10], 2, False),
    # Groups of 3, 1 and 2 rows plus one padding row.
    "dynamic": ([2, 2, 2, 0, 1, 1, PAD], [12, 13, 14, 8, 20, 21, 2], 3, True),
}


def decode_rows(req_ids, lengths, causal):
    """Per row: (request, keys the row sees causally, keys of the whole request)."""
    request_len = {}
    for r, n in zip(req_ids, lengths):
        request_len[r] = max(request_len.get(r, 0), n)
    return [(r, n, request_len[r] if not causal else n) for r, n in zip(req_ids, lengths)]


def decode_groups(req_ids, rows_per_request, dynamic):
    """Row indices of each attention request: fixed groups of rows, or runs of one request id with every
    padding row on its own."""
    if not dynamic:
        return [list(range(i, i + rows_per_request)) for i in range(0, len(req_ids), rows_per_request)]
    groups = []
    for i, r in enumerate(req_ids):
        if groups and r != PAD and req_ids[groups[-1][-1]] == r:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
def test_decode_layouts(layout, sinks, page_size):
    dtype, q_heads, kv_heads = torch.bfloat16, 8, 2
    req_ids, lengths, rows_per_request, dynamic = LAYOUTS[layout]
    token_map = paged_token_map(8, page_size, seed=2)
    kv, k, v = make_cache(kv_heads, dtype, seed=3)
    q = torch.randn(len(req_ids), q_heads, 64, device="cuda", dtype=dtype)
    sink_logits = torch.randn(q_heads, device="cuda", dtype=dtype) if sinks else None
    layer_config = GQALayerConfig(num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=dtype, has_sinks=sinks)
    batch = make_batch(
        token_map, req_ids, lengths, decode_rows_per_request=rows_per_request, dynamic_decode_rows=dynamic
    )
    backend = FA3GQADecodeBackend(layer_config)
    att_state = backend.create_att_state(batch, make_resources(page_size=page_size))
    out = torch.empty_like(q)
    assert backend.forward(att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v, sinks=sink_logits), out=out) is out
    keep = [i for i, r in enumerate(req_ids) if r != PAD]
    expected = reference(q, k, v, token_map, decode_rows(req_ids, lengths, True), sinks=sink_logits)
    torch.testing.assert_close(out[keep].float(), expected[keep], atol=0.02, rtol=0.02)


# With one row per request non-causal decode equals causal decode, so only multi-row layouts are checked.
@pytest.mark.parametrize("layout", ["fixed", "dynamic"])
@pytest.mark.parametrize("window", [(-1, -1), (6, 6)])
def test_decode_non_causal(layout, window):
    # Block-drafting models (DFlash) let every row of a request see all of the request's keys.
    dtype, q_heads, kv_heads, page_size = torch.bfloat16, 8, 2, 16
    req_ids, lengths, rows_per_request, dynamic = LAYOUTS[layout]
    token_map = paged_token_map(8, page_size, seed=2)
    _, k, v = make_cache(kv_heads, dtype, seed=3)
    generator = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(len(req_ids), q_heads, 64, device="cuda", dtype=dtype, generator=generator)
    layer_config = GQALayerConfig(
        num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=dtype, causal=False, sliding_window=window
    )
    batch = make_batch(
        token_map, req_ids, lengths, decode_rows_per_request=rows_per_request, dynamic_decode_rows=dynamic
    )
    backend = FA3GQADecodeBackend(layer_config)
    att_state = backend.create_att_state(batch, make_resources(page_size=page_size))
    out = torch.empty_like(q)
    backend.forward(att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v), out=out)
    keep = [i for i, r in enumerate(req_ids) if r != PAD]
    expected = reference(q, k, v, token_map, decode_rows(req_ids, lengths, False), causal=False, window=window)
    torch.testing.assert_close(out[keep].float(), expected[keep], atol=0.02, rtol=0.02)


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("static_q", [False, True])
def test_fp8_decode_layouts(layout, static_q):
    q_heads, kv_heads = 8, 2
    req_ids, lengths, rows_per_request, dynamic = LAYOUTS[layout]
    token_map = paged_token_map(8, 1, seed=2)
    k, v, k_ref, v_ref, k_scale, v_scale = make_fp8_cache(kv_heads, seed=3)
    generator = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(len(req_ids), q_heads, 64, device="cuda", generator=generator).bfloat16()
    q_scale = q.float().abs().view(q.shape[0], kv_heads, -1).amax(dim=(0, 2)) / FP8_MAX if static_q else None
    layer_config = GQALayerConfig(
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=64,
        dtype=torch.bfloat16,
        kv_quant="fp8_per_head",
        has_q_scale=static_q,
    )
    batch = make_batch(
        token_map, req_ids, lengths, decode_rows_per_request=rows_per_request, dynamic_decode_rows=dynamic
    )
    backend = FA3GQADecodeBackend(layer_config)
    att_state = backend.create_att_state(batch, make_resources(page_size=16))
    assert att_state.page_size == 1, "an FP8 cache runs with single-token pages"
    out = torch.empty_like(q)
    layer_inputs = GQALayerInputs(
        q=q, k_cache=k, v_cache=v, kv_scales=GQAKVScales(k_scale=k_scale, v_scale=v_scale, q_scale=q_scale)
    )
    assert backend.forward(att_state, layer_inputs, out=out) is out
    q_ref = fp8_round_q(q, kv_heads, decode_groups(req_ids, rows_per_request, dynamic), q_scale)
    keep = [i for i, r in enumerate(req_ids) if r != PAD]
    expected = reference(q_ref, k_ref, v_ref, token_map, decode_rows(req_ids, lengths, True))
    torch.testing.assert_close(out[keep].float(), expected[keep], atol=FP8_TOL, rtol=FP8_TOL)


def test_decode_keeps_graph_tensors_per_microbatch_and_batch_size():
    token_map = paged_token_map(8, 16, seed=4)
    backend = FA3GQADecodeBackend(GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16))
    resources = make_resources(page_size=16)
    req_ids = [2, 2, 0, 0]

    def build(lengths, microbatch_index=0, graph=True):
        return backend.create_att_state(
            make_batch(
                token_map,
                req_ids,
                lengths,
                decode_rows_per_request=2,
                cuda_graph_max_kv_seq_len=512 if graph else None,
                microbatch_index=microbatch_index,
            ),
            resources,
        )

    captured = build([33, 34, 9, 10])
    replayed = build([299, 300, 63, 64])
    for name in ("cache_seqlens", "cu_seqlens_q", "page_table"):
        assert getattr(replayed, name).data_ptr() == getattr(captured, name).data_ptr()
    assert captured.cache_seqlens.tolist() == [300, 64], "a replay rewrites the captured tensors in place"
    other = build([33, 34, 9, 10], microbatch_index=1)
    assert other.cache_seqlens.data_ptr() != captured.cache_seqlens.data_ptr()
    assert other.page_table.data_ptr() != captured.page_table.data_ptr()
    eager = build([33, 34, 9, 10], graph=False)
    assert eager.cache_seqlens.data_ptr() != captured.cache_seqlens.data_ptr()


def test_backend_rejects_unsupported_specs():
    with pytest.raises(NotImplementedError, match="FA3 GQA"):
        FA3GQADecodeBackend(GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=512, dtype=torch.bfloat16))


# ---- adapter through LightLLM's CudaGraph ----


@pytest.fixture
def start_args(monkeypatch):
    from lightllm.server.core.objs.start_args_type import StartArgs

    def configure(overlap: bool, **overrides) -> None:
        args = {f.name: getattr(StartArgs(), f.name) for f in dataclasses.fields(StartArgs)}
        args.update(enable_decode_microbatch_overlap=overlap, enable_torch_memory_saver=False, **overrides)
        monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(args))
        get_env_start_args.cache_clear()
        return args

    yield configure
    get_env_start_args.cache_clear()


def gqa_config(**fields) -> GQALayerConfig:
    return GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, **fields)


class _Model:
    """The model fields used during attention initialization; its layers run FA3."""

    def __init__(self, args, page_size: int, rows_per_request: int) -> None:
        backends = {"llm_prefill_att_backend": ["fa3"], "llm_decode_att_backend": ["fa3"]}
        self.args = SimpleNamespace(
            **{**args, **backends, "page_size": page_size, "running_max_req_size": 4, "mtp_step": 1}
        )
        self.graph_max_batch_size = 4
        self.max_seq_length = self.graph_max_len_in_batch = 512
        self.tp_world_size_ = 1
        self.is_mtp_draft_model = False
        self.rows_per_request = rows_per_request


def _decode_state(runner, token_map, req_ids, lengths, microbatch_index=0, *, capture=False, replay=False):
    s = InferStateInfo()
    s.is_prefill = False
    s.graph_mode = GraphMode.CAPTURE if capture else GraphMode.REPLAY if replay else GraphMode.EAGER
    s.cuda_graph_max_kv_seq_len = 512 if s.uses_cuda_graph else None
    s.decode_rows_per_request = runner.model.rows_per_request
    s.dynamic_decode_rows = runner.model.args.mtp_dynamic_verify
    s.input_ids = torch.zeros(len(lengths), device="cuda", dtype=torch.int64)
    s.batch_size, s.microbatch_index = len(lengths), microbatch_index
    s.b_req_idx = torch.tensor(req_ids, device="cuda", dtype=torch.int32)
    s.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    s.max_kv_seq_len, s.total_token_num = max(lengths), sum(lengths)
    s.b1_cu_q_seq_len = torch.arange(len(lengths) + 1, device="cuda", dtype=torch.int32)
    s.b1_cu_kv_seq_len = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    s.req_manager = SimpleNamespace(req_to_token_indexs=token_map, HOLD_REQUEST_ID=PAD)
    runner.step(s)
    return s


class _Microbatch:
    def __init__(self, runner, page_size, req_ids, seed):
        self.runner, self.req_ids = runner, req_ids
        self.token_map = paged_token_map(8, page_size, seed)
        _, self.k, self.v = make_cache(2, torch.bfloat16, seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.q = torch.randn(len(req_ids), 8, 64, device="cuda", dtype=torch.bfloat16, generator=generator)

    def state(self, lengths, microbatch_index, *, capture=False, replay=False):
        return _decode_state(
            self.runner, self.token_map, self.req_ids, lengths, microbatch_index, capture=capture, replay=replay
        )

    def forward(self, state):
        return self.runner.run_gqa(state, self.q, self.k, self.v)

    def check(self, actual, lengths):
        expected = reference(self.q, self.k, self.v, self.token_map, decode_rows(self.req_ids, lengths, True))
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "overlap,rows_per_request", [(False, 1), (True, 1), (False, 2)], ids=["normal", "overlap", "fixed_mtp"]
)
def test_layer_group_replays_through_cuda_graph(start_args, overlap, rows_per_request):
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    args = start_args(overlap=overlap)
    config = gqa_config()
    runner = GroupsRunner(_Model(args, 16, rows_per_request), config, config)
    graph = CudaGraph(4, 4, 4, max_batch_size=4, max_len_in_batch=512)
    if rows_per_request == 2:
        req_ids, steps = [2, 2, 0, 0], [(33, 34, 9, 10), (299, 300, 63, 64), (1, 2, 511, 512)]
    else:
        req_ids, steps = [2, 0, 1, 3], [(34, 10, 260, 3), (300, 512, 2, 77), (1, 1, 1, 1)]
    mbs = [_Microbatch(runner, 16, req_ids, seed) for seed in (5, 6)]
    if overlap:
        graph.capture_decode(
            lambda s0, s1: (mbs[0].forward(s0), mbs[1].forward(s1)),
            mbs[0].state(steps[0], 0, capture=True),
            mbs[1].state(steps[1], 1, capture=True),
        )
        for lengths0, lengths1 in [(steps[1], steps[2]), (steps[2], steps[0])]:
            out0, out1 = graph.replay(mbs[0].state(lengths0, 0, replay=True), mbs[1].state(lengths1, 1, replay=True))
            mbs[0].check(out0, lengths0)
            mbs[1].check(out1, lengths1)
    else:
        mb = mbs[0]
        graph.capture_decode(mb.forward, mb.state(steps[0], 0, capture=True))
        for lengths in steps[1:] + steps[:1]:
            mb.check(graph.replay(mb.state(lengths, 0, replay=True)), lengths)


# Dynamic verify steps of 6 rows and up to 3 rows per request: (b_req_idx, b_seq_len). After the captured
# step the grouping changes, a padding row appears, lengths grow up to the graph capacity and shrink back.
DYNAMIC_STEPS = [
    ([2, 2, 2, 0, 1, 1], [12, 13, 14, 8, 20, 21]),
    ([3, 0, 0, 0, 1, PAD], [300, 40, 41, 42, 7, 2]),
    ([4, 5, 6, 1, 1, 1], [3, 9, 511, 100, 101, 102]),
    ([2, 2, 2, 0, 1, 1], [12, 13, 14, 8, 20, 21]),
]


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8_per_head"])
def test_layer_group_replays_dynamic_verify_layout_through_cuda_graph(start_args, fp8):
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    args = start_args(overlap=False, mtp_dynamic_verify=True)
    model = _Model(args, 16, rows_per_request=3)
    config = gqa_config(kv_quant="fp8_per_head") if fp8 else gqa_config()
    runner = GroupsRunner(model, config, config)
    token_map = paged_token_map(8, 16, seed=7)
    generator = torch.Generator(device="cuda").manual_seed(9)
    q = torch.randn(6, 8, 64, device="cuda", generator=generator).bfloat16()
    if fp8:
        k, v, k_ref, v_ref, k_scale, v_scale = make_fp8_cache(2, seed=8)
        kv_scales, tol = GQAKVScales(k_scale=k_scale, v_scale=v_scale), FP8_TOL
    else:
        _, k, v = make_cache(2, torch.bfloat16, seed=8)
        k_ref, v_ref, kv_scales, tol = k, v, None, 0.02

    def forward(state):
        return runner.run_gqa(state, q, k, v, kv_scales=kv_scales)

    graph = CudaGraph(6, 6, 6, max_batch_size=6, max_len_in_batch=512)
    graph.capture_decode(forward, _decode_state(runner, token_map, *DYNAMIC_STEPS[0], capture=True))
    for req_ids, lengths in DYNAMIC_STEPS[1:]:
        out = graph.replay(_decode_state(runner, token_map, req_ids, lengths, replay=True))
        q_ref = fp8_round_q(q, 2, decode_groups(req_ids, 3, True)) if fp8 else q
        expected = reference(q_ref, k_ref, v_ref, token_map, decode_rows(req_ids, lengths, True))
        keep = [i for i, r in enumerate(req_ids) if r != PAD]
        torch.testing.assert_close(out[keep].float(), expected[keep], atol=tol, rtol=tol)
