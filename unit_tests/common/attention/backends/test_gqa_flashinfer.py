import dataclasses
import json
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention.backend import AttentionBatch
from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferGQABuilderConfig,
    FlashInferGQADecodeBackend,
    FlashInferGQAPrefillBackend,
)
from lightllm.common.basemodel.attention.base_att import AttControl, BaseAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.utils.envs_utils import get_env_start_args


flashinfer = pytest.importorskip("flashinfer")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def reference(q, k, v, token_map, req_ids, lengths, ready):
    outputs = []
    offset = 0
    for req_id, length, cached in zip(req_ids, lengths, ready):
        count = length - cached
        slots = token_map[req_id, :length].long()
        keys = k[slots].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1).transpose(0, 1)
        values = v[slots].float().repeat_interleave(q.shape[1] // v.shape[1], dim=1).transpose(0, 1)
        mask = torch.arange(length, device=q.device)[None, :] <= torch.arange(cached, length, device=q.device)[:, None]
        outputs.append(
            F.scaled_dot_product_attention(
                q[offset : offset + count].float().transpose(0, 1), keys, values, attn_mask=mask
            ).transpose(0, 1)
        )
        offset += count
    return torch.cat(outputs)


def make_batch(token_map, req_ids, lengths, query_lengths=None, *, graph_capacity=None, microbatch_index=0):
    lengths_t = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    query_lengths = query_lengths or [1] * len(lengths)
    cu_q = torch.tensor([0, *query_lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return AttentionBatch(
        req_to_token_indexs=token_map,
        b_req_idx=torch.tensor(req_ids, device="cuda", dtype=torch.int32),
        b_seq_len=lengths_t,
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        cu_q_seq_len=cu_q,
        max_q_seq_len=max(query_lengths),
        cuda_graph_max_kv_seq_len=graph_capacity,
        microbatch_index=microbatch_index,
    )


def paged_token_map(requests: int, page_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pages = torch.randperm(requests * 512 // page_size, device="cuda", dtype=torch.int32, generator=generator)
    pages = pages.view(requests, -1)
    return (pages[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(requests, 512)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("cached", [False, True])
def test_prefill_decode(dtype, q_heads, kv_heads, page_size, cached):
    torch.manual_seed(44)
    lengths, ready, req_ids = [33, 9, 259], ([16, 5, 128] if cached else [0, 0, 0]), [2, 0, 1]
    token_map = paged_token_map(3, page_size, seed=0)
    query_lengths = [n - r for n, r in zip(lengths, ready)]
    q = torch.randn(sum(query_lengths), q_heads, 64, device="cuda", dtype=dtype)
    kv = torch.randn(3 * 512, 2 * kv_heads, 64, device="cuda", dtype=dtype)
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]
    before, q_before = kv.clone(), q.clone()
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    config = FlashInferGQABuilderConfig(q_heads, kv_heads, 64, page_size, dtype, dtype)
    layer_config = GQALayerConfig(num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=dtype)

    step_builder = FlashInferGQAPrefillBackend.get_step_builder_cls()(
        config, workspace=lambda: workspace, max_seq_len=512
    )
    step_args = step_builder.build(make_batch(token_map, req_ids, lengths, query_lengths))
    out = torch.empty_like(q)
    impl = FlashInferGQAPrefillBackend.get_impl_cls()(layer_config)
    assert impl.forward(step_args, GQALayerInputs(q=q, k_cache=k, v_cache=v), out=out) is out
    torch.testing.assert_close(
        out.float(), reference(q, k, v, token_map, req_ids, lengths, ready), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(q, q_before, atol=0, rtol=0)

    step_builder = FlashInferGQADecodeBackend.get_step_builder_cls()(
        config, workspace=lambda: workspace, max_seq_len=512, max_batch_size=3, device=torch.device("cuda")
    )
    step_args = step_builder.build(make_batch(token_map, req_ids, lengths))
    q = torch.randn(3, q_heads, 64, device="cuda", dtype=dtype)
    out = torch.empty_like(q)
    impl = FlashInferGQADecodeBackend.get_impl_cls()(layer_config)
    assert impl.forward(step_args, GQALayerInputs(q=q, k_cache=k, v_cache=v), out=out) is out
    expected = reference(q, k, v, token_map, req_ids, lengths, [x - 1 for x in lengths])
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(kv, before, atol=0, rtol=0)


def test_decode_step_builder_keeps_one_graph_wrapper_per_microbatch_and_batch_size():
    token_map = paged_token_map(2, 16, seed=1)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    config = FlashInferGQABuilderConfig(8, 2, 64, 16, torch.bfloat16, torch.bfloat16)
    step_builder = FlashInferGQADecodeBackend.get_step_builder_cls()(
        config, workspace=lambda: workspace, max_seq_len=512, max_batch_size=2, device=torch.device("cuda")
    )
    captured = step_builder.build(make_batch(token_map, [0, 1], [33, 17], graph_capacity=512)).wrapper
    assert step_builder.build(make_batch(token_map, [0, 1], [300, 257], graph_capacity=512)).wrapper is captured
    other = step_builder.build(make_batch(token_map, [0, 1], [33, 17], graph_capacity=512, microbatch_index=1)).wrapper
    assert other is not captured
    eager = [step_builder.build(make_batch(token_map, [0, 1], [33, 17])).wrapper for _ in range(2)]
    assert eager[0] is not eager[1] and all(wrapper is not captured for wrapper in eager)


@pytest.mark.parametrize(
    "fields",
    [dict(sliding_window=(15, 0)), dict(has_sinks=True), dict(causal=False), dict(softmax_scale=1.0)],
)
def test_impls_reject_unsupported_specs(fields):
    layer_config = GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, **fields)
    for backend in (FlashInferGQAPrefillBackend, FlashInferGQADecodeBackend):
        with pytest.raises(NotImplementedError, match="FlashInfer GQA"):
            backend.get_impl_cls()(layer_config)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("causal,att_control", [(False, AttControl()), (True, AttControl(use_att_sink=True))])
def test_adapter_rejects_attention_flashinfer_cannot_run(phase, causal, att_control):
    # The adapter must forward causality and sinks into the layer config so the impl rejects them.
    from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferDecodeAttState, FlashInferPrefillAttState

    backend = SimpleNamespace(
        kv_quant=None,
        uses_causal_attention=lambda: causal,
        get_attention_impl=lambda backend_cls, layer_config: backend_cls.get_impl_cls()(layer_config),
    )
    owner = InferStateInfo()
    state_cls = FlashInferPrefillAttState if phase == "prefill" else FlashInferDecodeAttState
    state = state_cls(backend=backend, _infer_state_weak_ref=weakref.ref(owner))
    state.causal = backend.uses_causal_attention()
    q = torch.empty(2, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = v = torch.empty(16, 2, 64, device="cuda", dtype=torch.bfloat16)
    run = state.prefill_att if phase == "prefill" else state.decode_att
    with pytest.raises(NotImplementedError, match="FlashInfer GQA"):
        run(q=q, k=k, v=v, att_control=att_control)


# ---- adapter through LightLLM's CudaGraph ----


@pytest.fixture
def start_args(monkeypatch):
    from lightllm.server.core.objs.start_args_type import StartArgs

    def configure(overlap: bool) -> None:
        args = {f.name: getattr(StartArgs(), f.name) for f in dataclasses.fields(StartArgs)}
        args.update(enable_decode_microbatch_overlap=overlap, enable_torch_memory_saver=False)
        monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(args))
        get_env_start_args.cache_clear()

    yield configure
    get_env_start_args.cache_clear()


class _Model:
    """Hashable stub: BaseAttBackend keys its per-model singleton by the model object."""

    def __init__(self, page_size: int, dtype: torch.dtype) -> None:
        self.args = SimpleNamespace(page_size=page_size, flashinfer_workspace_size=64)
        self.config = {"num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512}
        self.max_seq_length = self.graph_max_len_in_batch = 512
        self.graph_max_batch_size = 2
        self.data_type = dtype
        self.is_mtp_draft_model = False


class _Microbatch:
    def __init__(self, backend, page_size, dtype, seed):
        self.backend = backend
        self.token_map = paged_token_map(2, page_size, seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        kv = torch.randn(1024, 4, 64, device="cuda", dtype=dtype, generator=generator)
        self.k, self.v = kv[:, :2], kv[:, 2:]
        self.q = torch.randn(2, 8, 64, device="cuda", dtype=dtype, generator=generator)

    def state(self, lengths, microbatch_index, *, capture=False, replay=False):
        state = InferStateInfo()
        state.is_prefill = False
        state.is_cuda_graph_capture, state.is_cuda_graph_replay = capture, replay
        state.input_ids = torch.zeros(2, device="cuda", dtype=torch.int64)
        state.batch_size, state.microbatch_index = 2, microbatch_index
        state.b_req_idx = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
        state.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        state.max_kv_seq_len, state.total_token_num = max(lengths), sum(lengths)
        state.req_manager = SimpleNamespace(req_to_token_indexs=self.token_map)
        state.decode_att_state = self.backend.create_att_decode_state(infer_state=state)
        state.init_att_state()
        return state

    def forward(self, state):
        return state.decode_att_state.decode_att(q=self.q, k=self.k, v=self.v)

    def check(self, actual, lengths):
        expected = reference(self.q, self.k, self.v, self.token_map, [1, 0], lengths, [x - 1 for x in lengths])
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
def test_adapter_replays_through_cuda_graph(start_args, overlap, page_size):
    from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    start_args(overlap=overlap)
    model = _Model(page_size, torch.bfloat16)
    backend = FlashInferAttBackend(model=model)
    try:
        graph = CudaGraph(2, 2, 2, max_batch_size=2, max_len_in_batch=512)
        mbs = [_Microbatch(backend, page_size, torch.bfloat16, seed) for seed in (2, 3)]
        if overlap:
            graph.capture_decode(
                lambda s0, s1: (mbs[0].forward(s0), mbs[1].forward(s1)),
                mbs[0].state((33, 17), 0, capture=True),
                mbs[1].state((5, 40), 1, capture=True),
            )
            # Microbatches replay with different lengths, so a shared wrapper or buffer would mix them up.
            for lengths0, lengths1 in [((129, 65), (2, 300)), ((300, 257), (64, 1))]:
                out0, out1 = graph.replay(
                    mbs[0].state(lengths0, 0, replay=True), mbs[1].state(lengths1, 1, replay=True)
                )
                mbs[0].check(out0, lengths0)
                mbs[1].check(out1, lengths1)
        else:
            mb = mbs[0]
            graph.capture_decode(mb.forward, mb.state((33, 17), 0, capture=True))
            for lengths in [(129, 65), (300, 257), (1, 2)]:
                mb.check(graph.replay(mb.state(lengths, 0, replay=True)), lengths)
    finally:
        BaseAttBackend._instances.pop((FlashInferAttBackend, model), None)
