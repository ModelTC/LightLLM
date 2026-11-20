import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backend import AttentionBatch
from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend, FA3GQAPrefillBackend
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.sgl_utils import flash_attn_with_kvcache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or flash_attn_with_kvcache is None, reason="FA3 (sgl_kernel) is required"
)
PAD = 7  # padding request id; its token-map row is valid but its output is ignored


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
    return AttentionBatch(
        req_to_token_indexs=token_map,
        b_req_idx=torch.tensor(req_ids, device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        cu_q_seq_len=cu_q,
        cu_kv_seq_len=cu_kv,
        max_q_seq_len=max(query_lengths),
        pad_req_idx=PAD,
        **kwargs,
    )


class _PageTables:
    """Caller-owned page-table storage with the view contract of Fa3AttBackend.get_page_table_view."""

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.storage = [torch.empty(64 * 512, dtype=torch.int32, device="cuda") for _ in range(2)]

    def __call__(self, rows: int, max_kv_len: int, microbatch_index: int) -> torch.Tensor:
        pages = (max_kv_len + self.page_size - 1) // self.page_size
        return self.storage[microbatch_index][: rows * pages].view(rows, pages)


def make_cache(kv_heads, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(8 * 512, 2 * kv_heads, 64, device="cuda", dtype=dtype, generator=generator)
    return kv, kv[:, :kv_heads], kv[:, kv_heads:]


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
    step_args = FA3GQAPrefillBackend.get_step_builder_cls()(page_size=page_size).build(batch)
    out = torch.empty_like(q)
    impl = FA3GQAPrefillBackend.get_impl_cls()(layer_config)
    assert impl.forward(step_args, GQALayerInputs(q=q, k_cache=k, v_cache=v, sinks=sink_logits), out=out) is out
    rows = [(r, pos + 1, n) for r, n, c in zip(req_ids, lengths, ready) for pos in range(c, n)]
    expected = reference(q, k, v, token_map, rows, causal=causal, window=window, sinks=sink_logits)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(kv, before, atol=0, rtol=0)


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
    step_builder = FA3GQADecodeBackend.get_step_builder_cls()(
        page_size=page_size, page_table_view=_PageTables(page_size)
    )
    step_args = step_builder.build(batch)
    out = torch.empty_like(q)
    impl = FA3GQADecodeBackend.get_impl_cls()(layer_config)
    assert impl.forward(step_args, GQALayerInputs(q=q, k_cache=k, v_cache=v, sinks=sink_logits), out=out) is out
    keep = [i for i, r in enumerate(req_ids) if r != PAD]
    expected = reference(q, k, v, token_map, decode_rows(req_ids, lengths, True), sinks=sink_logits)
    torch.testing.assert_close(out[keep].float(), expected[keep], atol=0.02, rtol=0.02)


def test_decode_step_builder_keeps_graph_tensors_per_microbatch_and_batch_size():
    token_map = paged_token_map(8, 16, seed=4)
    step_builder = FA3GQADecodeBackend.get_step_builder_cls()(page_size=16, page_table_view=_PageTables(16))
    req_ids = [2, 2, 0, 0]

    def build(lengths, microbatch_index=0, graph=True):
        return step_builder.build(
            make_batch(
                token_map,
                req_ids,
                lengths,
                decode_rows_per_request=2,
                cuda_graph_max_kv_seq_len=512 if graph else None,
                microbatch_index=microbatch_index,
            )
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


def test_impl_rejects_unsupported_specs():
    with pytest.raises(NotImplementedError, match="FA3 GQA"):
        FA3GQADecodeBackend.get_impl_cls()(
            GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=512, dtype=torch.bfloat16)
        )


# ---- adapter through LightLLM's CudaGraph ----


@pytest.fixture
def start_args(monkeypatch):
    from lightllm.server.core.objs.start_args_type import StartArgs

    def configure(overlap: bool) -> None:
        args = {f.name: getattr(StartArgs(), f.name) for f in dataclasses.fields(StartArgs)}
        args.update(enable_decode_microbatch_overlap=overlap, enable_torch_memory_saver=False)
        monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(args))
        get_env_start_args.cache_clear()
        return args

    yield configure
    get_env_start_args.cache_clear()


class _Model:
    """Hashable stub: BaseAttBackend keys its per-model singleton by the model object."""

    def __init__(self, args, page_size: int, rows_per_request: int) -> None:
        self.args = SimpleNamespace(**{**args, "page_size": page_size, "running_max_req_size": 4, "mtp_step": 1})
        self.graph_max_batch_size = 4
        self.max_seq_length = self.graph_max_len_in_batch = 512
        self.tp_world_size_ = 1
        self.is_mtp_draft_model = False
        self.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _: rows_per_request - 1)


class _Microbatch:
    def __init__(self, backend, page_size, req_ids, seed):
        self.backend, self.req_ids = backend, req_ids
        self.token_map = paged_token_map(8, page_size, seed)
        _, self.k, self.v = make_cache(2, torch.bfloat16, seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.q = torch.randn(len(req_ids), 8, 64, device="cuda", dtype=torch.bfloat16, generator=generator)

    def state(self, lengths, microbatch_index, *, capture=False, replay=False):
        s = InferStateInfo()
        s.is_prefill = False
        s.is_cuda_graph_capture, s.is_cuda_graph_replay = capture, replay
        s.input_ids = torch.zeros(len(lengths), device="cuda", dtype=torch.int64)
        s.batch_size, s.microbatch_index = len(lengths), microbatch_index
        s.b_req_idx = torch.tensor(self.req_ids, device="cuda", dtype=torch.int32)
        s.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        s.max_kv_seq_len, s.total_token_num = max(lengths), sum(lengths)
        s.b1_cu_q_seq_len = torch.arange(len(lengths) + 1, device="cuda", dtype=torch.int32)
        s.b1_cu_kv_seq_len = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        s.req_manager = SimpleNamespace(req_to_token_indexs=self.token_map, HOLD_REQUEST_ID=PAD)
        s.decode_att_state = self.backend.create_att_decode_state(infer_state=s)
        s.init_att_state()
        return s

    def forward(self, state):
        return state.decode_att_state.decode_att(q=self.q, k=self.k, v=self.v)

    def check(self, actual, lengths):
        expected = reference(self.q, self.k, self.v, self.token_map, decode_rows(self.req_ids, lengths, True))
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "overlap,rows_per_request", [(False, 1), (True, 1), (False, 2)], ids=["normal", "overlap", "fixed_mtp"]
)
def test_adapter_replays_through_cuda_graph(start_args, overlap, rows_per_request):
    from lightllm.common.basemodel.attention.fa3.fp import Fa3AttBackend
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    args = start_args(overlap=overlap)
    model = _Model(args, 16, rows_per_request)
    backend = Fa3AttBackend(model=model)
    try:
        graph = CudaGraph(4, 4, 4, max_batch_size=4, max_len_in_batch=512)
        if rows_per_request == 2:
            req_ids, steps = [2, 2, 0, 0], [(33, 34, 9, 10), (299, 300, 63, 64), (1, 2, 511, 512)]
        else:
            req_ids, steps = [2, 0, 1, 3], [(34, 10, 260, 3), (300, 512, 2, 77), (1, 1, 1, 1)]
        mbs = [_Microbatch(backend, 16, req_ids, seed) for seed in (5, 6)]
        if overlap:
            graph.capture_decode(
                lambda s0, s1: (mbs[0].forward(s0), mbs[1].forward(s1)),
                mbs[0].state(steps[0], 0, capture=True),
                mbs[1].state(steps[1], 1, capture=True),
            )
            for lengths0, lengths1 in [(steps[1], steps[2]), (steps[2], steps[0])]:
                out0, out1 = graph.replay(
                    mbs[0].state(lengths0, 0, replay=True), mbs[1].state(lengths1, 1, replay=True)
                )
                mbs[0].check(out0, lengths0)
                mbs[1].check(out1, lengths1)
        else:
            mb = mbs[0]
            graph.capture_decode(mb.forward, mb.state(steps[0], 0, capture=True))
            for lengths in steps[1:] + steps[:1]:
                mb.check(graph.replay(mb.state(lengths, 0, replay=True)), lengths)
    finally:
        BaseAttBackend._instances.pop((Fa3AttBackend, model), None)
