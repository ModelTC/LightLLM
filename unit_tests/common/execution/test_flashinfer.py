from dataclasses import replace

import pytest
import torch

from lightllm.core.compute.attention.dense.backend import DenseBatchMetadata, DenseStaticParams
from lightllm.core.compute.attention.dense.flashinfer import FlashInferAttentionBackend
from lightllm.core.planner.planner import AttentionRequest, Planner
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec, StateSpec, StateView


def test_distinct_formats_require_separate_plan_reservations(monkeypatch):
    monkeypatch.setattr("lightllm.core.compute.attention.dense.flashinfer.find_spec", lambda name: object())
    spec = DenseAttentionSpec(8, 2, 64, dtype=torch.float16)
    format = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)
    requests = [
        AttentionRequest(i, "decode", spec, StateSpec("main", f"kv.{i}", i, (i,), (2, 64)), fmt, None)
        for i, fmt in enumerate((format, replace(format, page_size=4)))
    ]
    planner = Planner(
        (FlashInferAttentionBackend("decode"),), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(200 * 1024 ** 2, 2, 768)
    )
    with pytest.raises(ValueError, match="plan workspace .* exceeds budget"):
        planner.prepare(requests)
    # Identical formats can still share one plan across serial layers.
    planner.prepare([replace(request, format=format) for request in requests])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_mixed_formats_keep_independent_plans_and_score_semantics(phase):
    torch.manual_seed(43)
    spec = DenseAttentionSpec(8, 2, 64, dtype=torch.float16, window=(127, 0), score_scale=0.2, softcap=2.0)
    plain = FormatSpec("plain", spec.dtype, "token_head_dim", ("k", "v"), 4)
    fp8 = FormatSpec(
        "fp8_per_tensor",
        torch.float8_e4m3fn,
        "token_head_dim",
        ("k", "v", "kv_scale", "q_scale"),
        4,
        scale_dtype=torch.float32,
    )
    allocate = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
    k, v = [torch.randn(1536, 2, 64, dtype=spec.dtype, device="cuda") for _ in range(2)]
    scales = torch.tensor([0.015625, 0.0078125], dtype=torch.float32, device="cuda")
    encoded = tuple((t.float() / scale).clamp(-448, 448).to(fp8.dtype) for t, scale in zip((k, v), scales))
    views = [
        StateView(StateSpec("main", f"kv.{i}", i, (i,), (2, 64)), fmt, fields)
        for i, (fmt, fields) in enumerate(((plain, (k, v)), (fp8, (*encoded, scales, None))))
    ]
    planner = Planner(
        (FlashInferAttentionBackend(phase),), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(512 * 1024 ** 2, 2, 768)
    )
    bindings = planner.bind(
        [(i, phase, spec, view, DenseStaticParams(spec, allocate), "auto") for i, view in enumerate(views)],
        lambda size: allocate(size, torch.uint8),
    ).bindings
    table = torch.arange(1536, device="cuda", dtype=torch.int32).view(2, 768)
    q_len = 3 if phase == "prefill" else 1
    q = torch.randn(2 * q_len, 8, 64, device="cuda", dtype=spec.dtype)

    def prepare(lengths):
        seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        cu_q = torch.arange(3, device="cuda", dtype=torch.int32) * q_len
        batch = DenseBatchMetadata(
            2,
            q_len,
            max(lengths),
            0,
            table,
            torch.arange(2, device="cuda", dtype=torch.int32),
            seq,
            cu_q[:-1],
            seq - q_len,
            cu_q,
            torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32),
        )
        return [b.execution.create_metadata(batch) for b in bindings]

    def check(outputs, lengths):
        caches = [(k.float(), v.float()), tuple(t.float() * scale for t, scale in zip(encoded, scales))]
        for output, (key, value) in zip(outputs, caches):
            reference = []
            for row, length in enumerate(lengths):
                query = q[row * q_len : (row + 1) * q_len].float().transpose(0, 1)
                keys = key[table[row, :length]].transpose(0, 1).repeat_interleave(4, dim=0)
                values = value[table[row, :length]].transpose(0, 1).repeat_interleave(4, dim=0)
                scores = spec.softcap * torch.tanh((query @ keys.transpose(-1, -2)) * spec.scale / spec.softcap)
                distance = torch.arange(length - q_len, length, device="cuda")[:, None] - torch.arange(
                    length, device="cuda"
                )
                scores.masked_fill_((distance < 0) | (distance > spec.window[0]), float("-inf"))
                reference.append((scores.softmax(-1) @ values).transpose(0, 1))
            torch.testing.assert_close(output.float(), torch.cat(reference), rtol=2e-2, atol=5e-3)

    lengths = [513, 65]
    metadata = prepare(lengths)
    check([b.execution.run(q, meta) for b, meta in zip(bindings, metadata)], lengths)
    graphs, outputs = [], []
    for binding, meta in zip(bindings, metadata):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = binding.execution.run(q, meta)
        graphs.append(graph)
        outputs.append(output)
    lengths = [517, 68]
    for old, new in zip(metadata, prepare(lengths)):
        old.copy_for_graph(new)
    q.add_(0.1)
    for graph in graphs:
        graph.replay()
    check(outputs, lengths)
