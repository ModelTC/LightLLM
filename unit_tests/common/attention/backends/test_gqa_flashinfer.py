from typing import Any
from unit_tests.common.attention.backends.step_fixture import make_backend_model
from unit_tests.common.attention.backends.step_fixture import create_att_state
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.attention.backends.gqa import GQAKVScales, GQAAttConfig, GQAAttInputs
from lightllm.common.attention.backends.gqa.selector import select_gqa_prefill_backend, select_gqa_decode_backend
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferGQADecodeBackend,
    FlashInferGQAPrefillBackend,
)
from unit_tests.common.attention.backends.groups_runner import GroupsRunner
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
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
    return make_infer_state(
        req_to_token_indexs=token_map,
        b_req_idx=torch.tensor(req_ids, device="cuda", dtype=torch.int32),
        b_seq_len=lengths_t,
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        b1_cu_q_seq_len=cu_q,
        max_q_seq_len=max(query_lengths),
        cuda_graph_max_kv_seq_len=graph_capacity,
        microbatch_index=microbatch_index,
    )


def paged_token_map(requests: int, page_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pages = torch.randperm(requests * 512 // page_size, device="cuda", dtype=torch.int32, generator=generator)
    pages = pages.view(requests, -1)
    return (pages[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(requests, 512)


def prefill_backend(config: Any, page_size: Any):
    init = FlashInferGQAPrefillBackend.RuntimeConfig(
        page_size=page_size, max_seq_len=512, workspace_bytes=32 * 1024 * 1024
    )
    return FlashInferGQAPrefillBackend(att_config=config, runtime_config=init)


def decode_backend(config: Any, page_size: Any, max_batch_size: Any):
    init = FlashInferGQADecodeBackend.RuntimeConfig(
        page_size=page_size, max_seq_len=512, max_batch_size=max_batch_size, workspace_bytes=32 * 1024 * 1024
    )
    return FlashInferGQADecodeBackend(att_config=config, runtime_config=init)


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
    att_config = GQAAttConfig(num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=dtype)

    impl = prefill_backend(config=att_config, page_size=page_size)
    att_state = create_att_state(backend=impl, infer_state=make_batch(token_map, req_ids, lengths, query_lengths))
    out = torch.empty_like(q)
    assert impl.prefill(att_state, GQAAttInputs(q=q, k_cache=k, v_cache=v), out=out) is out
    torch.testing.assert_close(
        out.float(), reference(q, k, v, token_map, req_ids, lengths, ready), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(q, q_before, atol=0, rtol=0)

    impl = decode_backend(config=att_config, page_size=page_size, max_batch_size=2)
    att_state = create_att_state(backend=impl, infer_state=make_batch(token_map, req_ids, lengths))
    q = torch.randn(3, q_heads, 64, device="cuda", dtype=dtype)
    out = torch.empty_like(q)
    assert impl.decode(att_state, GQAAttInputs(q=q, k_cache=k, v_cache=v), out=out) is out
    expected = reference(q, k, v, token_map, req_ids, lengths, [x - 1 for x in lengths])
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(kv, before, atol=0, rtol=0)


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("page_size", [1, 16])
def test_fp8_prefill_decode(q_heads, kv_heads, page_size):
    # A per-tensor FP8 cache with host-float scales; FlashInfer dequantizes K/V in the kernel and keeps q in
    # BF16, so against the dequantized cache only BF16 rounding remains and the BF16 tolerance applies.
    lengths, ready, req_ids = [33, 9, 259], [16, 5, 128], [2, 0, 1]
    token_map = paged_token_map(3, page_size, seed=0)
    generator = torch.Generator(device="cuda").manual_seed(1)
    kv = torch.randn(3 * 512, 2, kv_heads, 64, device="cuda", generator=generator)
    # Different K and V scales, so swapping them changes the result; |kv| / scale stays inside E4M3's 448.
    k_scale, v_scale = 0.02, 0.035
    encoded = (kv / torch.tensor([k_scale, v_scale], device="cuda")[None, :, None, None]).to(torch.float8_e4m3fn)
    k, v = encoded[:, 0].view(torch.uint8), encoded[:, 1].view(torch.uint8)
    k_ref, v_ref = encoded[:, 0].float() * k_scale, encoded[:, 1].float() * v_scale
    att_config = GQAAttConfig(
        num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=torch.bfloat16, kv_quant="fp8_per_tensor"
    )
    scales = GQAKVScales(k_scale=k_scale, v_scale=v_scale)

    query_lengths = [n - r for n, r in zip(lengths, ready)]
    q = torch.randn(sum(query_lengths), q_heads, 64, device="cuda", generator=generator).bfloat16()
    impl = prefill_backend(config=att_config, page_size=page_size)
    att_state = create_att_state(backend=impl, infer_state=make_batch(token_map, req_ids, lengths, query_lengths))
    out = torch.empty_like(q)
    impl.prefill(att_state, GQAAttInputs(q=q, k_cache=k, v_cache=v, kv_scales=scales), out=out)
    expected = reference(q, k_ref, v_ref, token_map, req_ids, lengths, ready)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)

    impl = decode_backend(config=att_config, page_size=page_size, max_batch_size=3)
    att_state = create_att_state(backend=impl, infer_state=make_batch(token_map, req_ids, lengths))
    q = torch.randn(3, q_heads, 64, device="cuda", generator=generator).bfloat16()
    out = torch.empty_like(q)
    impl.decode(att_state, GQAAttInputs(q=q, k_cache=k, v_cache=v, kv_scales=scales), out=out)
    expected = reference(q, k_ref, v_ref, token_map, req_ids, lengths, [x - 1 for x in lengths])
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)


def test_decode_wrapper_belongs_to_the_capture_state():
    import weakref

    pytest.importorskip("flashinfer")
    token_map = paged_token_map(requests=2, page_size=16, seed=1)
    config = GQAAttConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16)
    builder = decode_backend(config=config, page_size=16, max_batch_size=2)
    capture_batch = make_batch(token_map, [0, 1], [33, 17], graph_capacity=512)
    captured = create_att_state(backend=builder, infer_state=capture_batch)
    wrapper_ref = weakref.ref(captured.wrapper)
    replay_batch = make_batch(token_map, [0, 1], [300, 257], graph_capacity=512)
    replay_batch.graph_mode = GraphMode.REPLAY
    replay = create_att_state(backend=builder, infer_state=replay_batch)
    assert replay.wrapper is None
    capture_batch.copy_for_cuda_graph(new_infer_state=replay_batch)
    assert captured.wrapper is wrapper_ref()
    del captured, capture_batch
    assert wrapper_ref() is None


@pytest.mark.parametrize(
    "fields",
    [dict(sliding_window=(15, 0)), dict(has_sinks=True), dict(causal=False), dict(softmax_scale=1.0)],
)
def test_impls_reject_unsupported_specs(fields):
    att_config = GQAAttConfig(
        num_heads=fields.get("num_heads", 8),
        num_kv_heads=fields.get("num_kv_heads", 2),
        head_dim=fields.get("head_dim", 64),
        dtype=fields.get("dtype", torch.bfloat16),
        sliding_window=fields.get("sliding_window", (-1, -1)),
        softmax_scale=fields.get("softmax_scale", None),
        causal=fields.get("causal", True),
        has_sinks=fields.get("has_sinks", False),
        kv_quant=fields.get("kv_quant", None),
        kv_quant_group_size=fields.get("kv_quant_group_size", None),
        has_q_scale=fields.get("has_q_scale", False),
        has_alibi=fields.get("has_alibi", False),
        bidirectional_image_spans=fields.get("bidirectional_image_spans", False),
    )
    for backend in (FlashInferGQAPrefillBackend, FlashInferGQADecodeBackend):
        with pytest.raises(NotImplementedError, match="FlashInfer GQA"):
            backend(att_config)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("fields", [{"causal": False}, {"has_sinks": True}])
def test_layers_flashinfer_cannot_run_are_rejected_when_selected(phase, fields):
    att_config = GQAAttConfig(
        num_heads=fields.get("num_heads", 8),
        num_kv_heads=fields.get("num_kv_heads", 2),
        head_dim=fields.get("head_dim", 64),
        dtype=fields.get("dtype", torch.bfloat16),
        sliding_window=fields.get("sliding_window", (-1, -1)),
        softmax_scale=fields.get("softmax_scale", None),
        causal=fields.get("causal", True),
        has_sinks=fields.get("has_sinks", False),
        kv_quant=fields.get("kv_quant", None),
        kv_quant_group_size=fields.get("kv_quant_group_size", None),
        has_q_scale=fields.get("has_q_scale", False),
        has_alibi=fields.get("has_alibi", False),
        bidirectional_image_spans=fields.get("bidirectional_image_spans", False),
    )
    with pytest.raises(ValueError, match="FlashInfer GQA"):
        select = select_gqa_prefill_backend if phase == "prefill" else select_gqa_decode_backend
        select(att_config=att_config, requested="flashinfer", model=make_backend_model(), logical_layer_ids=(0,))


# 通过 LightLLM 的 CudaGraph 测试层组


@pytest.fixture
def start_args(monkeypatch):
    from lightllm.server.core.objs.start_args_type import StartArgs

    def configure(overlap: bool) -> dict:
        args = {f.name: getattr(StartArgs(), f.name) for f in dataclasses.fields(StartArgs)}
        args.update(enable_decode_microbatch_overlap=overlap, enable_torch_memory_saver=False)
        monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(args))
        get_env_start_args.cache_clear()
        return args

    yield configure
    get_env_start_args.cache_clear()


class _Model:
    """The model fields used during attention initialization; its layers run FlashInfer."""

    def __init__(self, args, page_size: int) -> None:
        backends = {"att_backend": {"gqa.prefill": "flashinfer", "gqa.decode": "flashinfer"}}
        self.args = SimpleNamespace(
            **{**args, **backends, "page_size": page_size, "flashinfer_workspace_size": 64, "running_max_req_size": 2}
        )
        self.max_seq_length = self.graph_max_len_in_batch = 512
        self.graph_max_batch_size = 2
        self.tp_world_size_ = 1
        self.is_mtp_draft_model = False
        self.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda is_draft_model: 0)


class _Microbatch:
    def __init__(self, runner, page_size, dtype, seed):
        self.runner = runner
        self.token_map = paged_token_map(2, page_size, seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        kv = torch.randn(1024, 4, 64, device="cuda", dtype=dtype, generator=generator)
        self.k, self.v = kv[:, :2], kv[:, 2:]
        self.q = torch.randn(2, 8, 64, device="cuda", dtype=dtype, generator=generator)

    def state(self, lengths, microbatch_index, *, capture=False, replay=False):
        state = InferStateInfo()
        state.is_prefill = False
        state.graph_mode = GraphMode.CAPTURE if capture else GraphMode.REPLAY if replay else GraphMode.EAGER
        state.input_ids = torch.zeros(2, device="cuda", dtype=torch.int64)
        state.batch_size, state.microbatch_index = 2, microbatch_index
        state.b_req_idx = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
        state.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        state.max_kv_seq_len, state.total_token_num = max(lengths), sum(lengths)
        state.req_manager = SimpleNamespace(req_to_token_indexs=self.token_map)
        self.runner.step(state)
        return state

    def forward(self, state):
        return self.runner.run_gqa(state, self.q, self.k, self.v)

    def check(self, actual, lengths):
        expected = reference(self.q, self.k, self.v, self.token_map, [1, 0], lengths, [x - 1 for x in lengths])
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
def test_layer_group_replays_through_cuda_graph(start_args, overlap, page_size):
    from lightllm.common.basemodel.cuda_graph import CudaGraph

    config = GQAAttConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16)
    runner = GroupsRunner(_Model(start_args(overlap=overlap), page_size), config, config)
    graph = CudaGraph(2, 2, 2, max_batch_size=2, max_len_in_batch=512)
    mbs = [_Microbatch(runner, page_size, torch.bfloat16, seed) for seed in (2, 3)]
    if overlap:
        graph.capture_decode(
            lambda s0, s1: (mbs[0].forward(s0), mbs[1].forward(s1)),
            mbs[0].state((33, 17), 0, capture=True),
            mbs[1].state((5, 40), 1, capture=True),
        )
        # Microbatches replay with different lengths, so a shared wrapper or buffer would mix them up.
        for lengths0, lengths1 in [((129, 65), (2, 300)), ((300, 257), (64, 1))]:
            out0, out1 = graph.replay(mbs[0].state(lengths0, 0, replay=True), mbs[1].state(lengths1, 1, replay=True))
            mbs[0].check(out0, lengths0)
            mbs[1].check(out1, lengths1)
    else:
        mb = mbs[0]
        graph.capture_decode(mb.forward, mb.state((33, 17), 0, capture=True))
        for lengths in [(129, 65), (300, 257), (1, 2)]:
            mb.check(graph.replay(mb.state(lengths, 0, replay=True)), lengths)
