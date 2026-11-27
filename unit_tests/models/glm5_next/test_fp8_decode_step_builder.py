"""GLM-5 FP8 sparse decode through the serving path: InferStateInfo -> decode backend build.

Unlike test_fp8_kv.py, the attention state here come from the layer group's backend, so these tests cover how
its build marks CUDA graph steps (which picks the direct Triton decode for small graph workloads) and keeps
FlashMLA's scheduler metadata per (microbatch, batch size) for replays.
"""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.backends.sparse_mla.flashmla import FlashMLASparseDecodeBackend
from lightllm.common.attention_state.full_att.packed_mla.fp8_e8m0_packed_mla_write import write_fp8_e8m0_packed_kv
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo
from lightllm.common.basemodel.triton_kernel.gen_decode_params import gen_decode_params

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 9,
        reason="Validated FlashMLA sparse decode target is Hopper",
    ),
]

SLOTS, TOPK_WIDTH, SCALE = 8192, 2176, 0.0625


def make_backend():
    """A GLM-5 FP8 layer group's decode backend."""
    pytest.importorskip("flash_mla")
    from unit_tests.models.glm5_next.test_fp8_kv import _layer_config

    config = _layer_config(torch.empty(1, 64, 512, dtype=torch.bfloat16), SCALE)
    return SimpleNamespace(
        decode_backend=FlashMLASparseDecodeBackend(config),
    )


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def make_infer(lengths, *, capture=False, replay=False, microbatch=0):
    infer = InferStateInfo()
    infer.is_prefill = False
    infer.batch_size = len(lengths)
    infer.b_req_idx = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    infer.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    infer.max_kv_seq_len, infer.total_token_num = max(lengths), sum(lengths)
    infer.max_q_seq_len = 1
    infer.req_manager = SimpleNamespace(req_to_token_indexs=None, HOLD_REQUEST_ID=len(lengths) + 1)
    infer.graph_mode = GraphMode.CAPTURE if capture else GraphMode.REPLAY if replay else GraphMode.EAGER
    infer.cuda_graph_max_kv_seq_len = SLOTS if infer.uses_cuda_graph else None
    infer.microbatch_index = microbatch
    (
        infer.b_q_seq_len,
        infer.b1_cu_q_seq_len,
        infer.b_kv_seq_len,
        infer.b1_cu_kv_seq_len,
        infer.position_ids,
    ) = gen_decode_params(infer.b_seq_len)
    return infer


def init_decode_state(backend, infer):
    """Prepare the attention state of one decode microbatch."""
    from unit_tests.models.glm5_next.test_fp8_kv import _decode

    att_state = backend.decode_backend.build(infer)
    return SimpleNamespace(att_state=att_state, decode_att=lambda q, kv, params: _decode(att_state, q, kv, params))


def select(lengths, out, seed):
    # Whole kpool=4 pools within the top-k width plus the newest partial pool: at most 2051 valid entries.
    out.fill_(-1)
    for row, length in enumerate(lengths):
        count = min(length // 4 * 4, 2048) + length % 4
        out[row, :count] = torch.randperm(SLOTS, device="cuda", generator=gen(seed + row))[:count].int()
    return out


def dequantize(packed):
    values = packed[..., :448].view(torch.float8_e4m3fn).float().reshape(-1, 7, 64)
    scales = packed[..., 576:583].view(torch.float8_e8m0fnu).float().reshape(-1, 7, 1)
    tail = packed[..., 448:576].view(torch.bfloat16).float().reshape(-1, 64)
    return torch.cat(((values * scales).reshape(-1, 448), tail), dim=-1)


def reference(q, cache, indices):
    out = []
    for row, ids in enumerate(indices):
        keys = cache[ids[ids >= 0].long()]
        out.append((q[row].float() @ keys.T * SCALE).softmax(-1) @ keys)
    return torch.stack(out)


def capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


@pytest.mark.parametrize(
    "batch, heads, direct",
    [(4, 64, False), (2, 16, True)],
    ids=["flashmla_graph", "direct_graph"],
)
def test_fp8_decode_graph_replays_steps_built_by_the_backend(monkeypatch, batch, heads, direct):
    from lightllm.common.attention.kernels.sparse_mla import fp8_group64_decode

    backend = make_backend()
    direct_calls = []
    direct_decode = fp8_group64_decode.fp8_sparse_direct_decode

    def counting_direct_decode(*args, **kwargs):
        direct_calls.append(torch.cuda.is_current_stream_capturing())
        return direct_decode(*args, **kwargs)

    monkeypatch.setattr(fp8_group64_decode, "fp8_sparse_direct_decode", counting_direct_decode)

    packed = torch.zeros(SLOTS, 1, 720, device="cuda", dtype=torch.uint8)
    source = torch.randn(SLOTS, 1, 512, device="cuda", dtype=torch.bfloat16, generator=gen(0))
    write_fp8_e8m0_packed_kv(source, torch.arange(SLOTS, device="cuda", dtype=torch.int32), packed)
    cache = dequantize(packed)
    first = [100, 2051, 7, 4096][:batch]
    second = [5000, 2050, 33, 3][:batch]
    indices = select(first, torch.empty((batch, TOPK_WIDTH), device="cuda", dtype=torch.int32), seed=10)
    q = torch.randn(batch, heads, 512, device="cuda", dtype=torch.bfloat16, generator=gen(1))
    control = {"topk_mem_indices": indices, "softmax_scale": SCALE, "kv_lora_rank": 512}

    def run(state):
        return state.decode_att((q, q[..., :0]), packed[..., :584], control)

    def eager(lengths):
        state = init_decode_state(backend, make_infer(lengths))
        assert not state.att_state.uses_cuda_graph and backend.decode_backend.direct_decode_bound == 2051
        return run(state).clone()

    eager_first = eager(first)
    torch.testing.assert_close(eager_first.float(), reference(q, cache, indices), atol=0.012, rtol=0.015)
    # The direct kernel serves only small CUDA graph workloads; eager steps keep native FlashMLA.
    assert direct_calls == []

    graph_infer = make_infer(first, capture=True)
    graph_state = init_decode_state(backend, graph_infer)
    assert graph_state.att_state.uses_cuda_graph
    graph, graph_out = capture(lambda: run(graph_state))
    assert (True in direct_calls) == direct
    graph.replay()
    torch.testing.assert_close(graph_out.float(), reference(q, cache, indices), atol=0.012, rtol=0.015)

    select(second, indices, seed=20)
    q.normal_(generator=gen(2))
    replay_infer = make_infer(second, replay=True)
    replay_state = init_decode_state(backend, replay_infer)
    # A replay step's state carries the scheduler metadata the graph captured.
    assert replay_state.att_state.sched_meta is graph_state.att_state.sched_meta
    graph_infer.copy_for_cuda_graph(replay_infer)
    graph.replay()
    replayed = graph_out.clone()
    torch.testing.assert_close(replayed.float(), reference(q, cache, indices), atol=0.012, rtol=0.015)
    if not direct:
        assert torch.equal(replayed, eager(second))


def test_fp8_decode_scheduler_metadata_is_kept_per_microbatch_and_batch_size():
    backend = make_backend()

    def sched_meta(lengths, **kwargs):
        return init_decode_state(backend, make_infer(lengths, **kwargs)).att_state.sched_meta

    captured = sched_meta([9, 10], capture=True)
    assert sched_meta([3000, 1], replay=True) is captured
    # Microbatch overlap captures one graph per microbatch, and every batch size has its own graph.
    other_microbatch = sched_meta([9, 10], capture=True, microbatch=1)
    other_batch_size = sched_meta([9, 10, 11], capture=True)
    assert len({id(captured), id(other_microbatch), id(other_batch_size)}) == 3
    assert sched_meta([3000, 1], replay=True, microbatch=1) is other_microbatch
    # Eager steps plan their own metadata on every step.
    eager = sched_meta([9, 10])
    assert eager is not captured and sched_meta([9, 10]) is not eager
