"""Small real-kernel checks; no checkpoint, service or distributed group required."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.models.llama.model import LlamaTpPartModel
from unit_tests.common.execution.test_model_binding import make_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Batch(SimpleNamespace):
    pass


@pytest.mark.parametrize("implementation", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_bound_attention_matches_sdpa_and_graph(phase, implementation, monkeypatch):
    torch.manual_seed(17)
    model, state = make_model(LlamaTpPartModel, "cuda", implementation)
    model._init_execution_plan()
    layer = model.layers_infer[0]
    # Remove runtime access to global resources after binding.
    model.att_state_manager = None
    indices = torch.tensor([0, 1, 2, 4, 5], device="cuda", dtype=torch.int32)
    kv = torch.randn(5, 4, 64, device="cuda", dtype=torch.float16)
    batch = Batch(
        batch_size=2,
        max_kv_seq_len=3,
        max_q_seq_len=3,
        total_token_num=5,
        b_req_idx=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([3, 2], device="cuda", dtype=torch.int32),
        b_q_start_loc=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
        b_ready_cache_len=torch.zeros(2, device="cuda", dtype=torch.int32),
        mem_index=indices,
        req_manager=SimpleNamespace(
            req_to_token_indexs=torch.tensor([[0, 1, 2], [4, 5, 0]], device="cuda", dtype=torch.int32)
        ),
    )
    batch.input_ids = torch.zeros(5, device="cuda", dtype=torch.int64)
    batch.b1_cu_q_seq_len = torch.tensor(
        [0, 3, 5] if phase == "prefill" else [0, 1, 2], device="cuda", dtype=torch.int32
    )
    batch.b1_cu_kv_seq_len = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    batch.microbatch_index = 0
    layer._post_cache_kv(kv, batch, None)
    batch.is_prefill = phase == "prefill"
    batch.execution_metadata = model.execution_runner.prepare_batch(batch)
    q = torch.randn(5 if phase == "prefill" else 2, 4, 64, device="cuda", dtype=torch.float16)
    bound = getattr(layer, f"bound_{phase}_attention")
    metadata = bound.prepare_batch(batch)
    output = bound.run(q, metadata)
    expected = []
    for row, start, length in [(0, 0, 3), (1, 3, 2)]:
        query = q[start : start + length] if phase == "prefill" else q[row : row + 1]
        key, value = kv[start : start + length, :2], kv[start : start + length, 2:]
        result = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            is_causal=phase == "prefill",
            enable_gqa=True,
        )
        expected.append(result.squeeze(0).transpose(0, 1))
    torch.testing.assert_close(output, torch.cat(expected), rtol=3e-3, atol=3e-3)
    torch.cuda.synchronize()
    # Warm the kernel on a side stream before capture, then replay with new Q values.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            bound.run(q, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound.run(q, metadata)
    q.add_(0.125)
    graph.replay()
    torch.testing.assert_close(captured, bound.run(q, metadata), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("implementation", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads", [2, 8])
@pytest.mark.parametrize("phase", ["prefill", "decode", "verify"])
def test_paged_long_sequences_and_concurrent_lanes(implementation, dtype, q_heads, phase):
    from lightllm.common.execution import (
        DenseAttentionSpec,
        DeviceCaps,
        ExecutionLimits,
        FormatSpec,
        Planner,
        StateSpec,
        StateView,
    )
    from lightllm.core.compute.attention.dense.backend import DenseAttentionBackend, DenseStaticParams
    from lightllm.core.compute.attention.dense.flashinfer import FlashInferAttentionBackend
    from lightllm.common.execution.runner import attention_batch

    torch.manual_seed(29)
    spec = DenseAttentionSpec(q_heads, 2, 64, dtype=dtype)
    format = FormatSpec("plain", dtype, "token_head_dim", ("k", "v"), 4)
    k, v = [torch.randn(2304, 2, 64, dtype=dtype, device="cuda") for _ in range(2)]
    view = StateView(StateSpec("main", "kv", 0, (0,), (2, 64)), format, (k, v))
    limits = ExecutionLimits((512 if implementation == "flashinfer" else 8) * 1024 * 1024, 3, 768, 2)
    backend = (
        FlashInferAttentionBackend(phase)
        if implementation == "flashinfer"
        else DenseAttentionBackend(implementation, phase)
    )
    planner = Planner((backend,), DeviceCaps("cuda", 0, (9, 0)), limits)
    params = DenseStaticParams(spec, lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"))
    bound = (
        planner.bind(
            [(0, phase, spec, view, params, "auto")], lambda size: torch.empty(size, dtype=torch.uint8, device="cuda")
        )
        .bindings[0]
        .execution
    )
    table = torch.arange(2304, device="cuda", dtype=torch.int32).view(3, 768)
    q_len = 2 if phase == "prefill" else 1

    def prepare(lengths, lane, rows):
        batch = Batch(
            batch_size=3,
            max_q_seq_len=q_len,
            max_kv_seq_len=max(lengths),
            microbatch_index=lane,
            req_manager=SimpleNamespace(req_to_token_indexs=table, HOLD_REQUEST_ID=2),
            b_req_idx=torch.tensor(rows, device="cuda", dtype=torch.int32),
            b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            b_q_start_loc=torch.arange(3, device="cuda", dtype=torch.int32) * q_len,
            b_ready_cache_len=torch.tensor(lengths, device="cuda", dtype=torch.int32) - q_len,
            b1_cu_q_seq_len=torch.arange(4, device="cuda", dtype=torch.int32) * q_len,
            b1_cu_kv_seq_len=torch.tensor([0] + lengths, device="cuda", dtype=torch.int32).cumsum(0),
        )
        return bound.create_metadata(attention_batch(batch))

    def reference(q, lengths, rows):
        result = []
        for index, (length, row) in enumerate(zip(lengths, rows)):
            query = q[index * q_len : (index + 1) * q_len]
            key, value = k[row * 768 : row * 768 + length], v[row * 768 : row * 768 + length]
            mask = (
                torch.arange(length, device="cuda")[None, :]
                <= (length - q_len + torch.arange(q_len, device="cuda"))[:, None]
            )
            output = F.scaled_dot_product_attention(
                query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), attn_mask=mask, enable_gqa=True
            )
            result.append(output.transpose(0, 1))
        return torch.cat(result)

    lengths, rows = [257, 513, 7], [0, 1, 2]
    metadata = [prepare(lengths, lane, rows) for lane in range(2)]
    queries = [torch.randn(3 * q_len, q_heads, 64, device="cuda", dtype=dtype) for _ in range(2)]
    for q, meta in zip(queries, metadata):
        torch.testing.assert_close(bound.run(q, meta), reference(q, lengths, rows), atol=5e-3, rtol=1e-2)
    streams = [torch.cuda.Stream() for _ in range(2)]
    outputs = []
    for stream, q, meta in zip(streams, queries, metadata):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(bound.run(q, meta))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for q, output in zip(queries, outputs):
        torch.testing.assert_close(output, reference(q, lengths, rows), atol=5e-3, rtol=1e-2)
    if phase != "prefill" or implementation == "flashinfer":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound.run(queries[0], metadata[0])
        lengths, rows = [515, 383, 9], [1, 0, 2]
        new_metadata = prepare(lengths, 0, rows)
        metadata[0].copy_for_graph(new_metadata)
        graph.replay()
        torch.testing.assert_close(captured, reference(queries[0], lengths, rows), atol=5e-3, rtol=1e-2)
