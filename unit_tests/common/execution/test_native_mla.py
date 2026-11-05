"""Latent MLA binding, explicit TP replicas and independent attention references."""

from lightllm.common.execution.formats.kv.spec import MLAKVStateSpec


from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.execution.contracts.device import DeviceCaps
from lightllm.common.execution.contracts.device import ExecutionLimits
from lightllm.common.execution.contracts.state import FormatSpec
from lightllm.common.execution.compute.attention.mla.spec import MLAAttentionSpec
from lightllm.common.execution.planner.planner import Planner
from lightllm.common.execution.contracts.state import StateSpec
from lightllm.common.execution.compute.attention.batch import AttentionBatch
from lightllm.common.execution.compute.attention.mla.backend import MLAAttentionBackend, MLAStaticParams
from lightllm.common.execution.formats.kv.mla import MLAKVWriter
from lightllm.common.execution.compute.attention.mla.flashinfer import FlashInferMLABackend
from lightllm.common.execution.storage.attention import MLAStateStore
from lightllm.common.execution.planner.planner import AttentionRequest


def test_mla_declarations_and_tp_replicas():
    from lightllm.models.deepseek2.adapter import Deepseek2ModelAdapter

    config = dict(
        num_hidden_layers=2,
        num_attention_heads=16,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_scaling={"factor": 8, "mscale_all_dim": 1},
    )
    adapter = Deepseek2ModelAdapter(config, role="main", dtype=torch.bfloat16, tp_world_size=4)
    spec = adapter.attention_specs()[0]
    assert spec.q_heads == 4 and spec.scale > 192 ** -0.5
    assert all(s.params == MLAKVStateSpec(latent_dim=512, rope_dim=64) for s in adapter.state_specs())
    fmt = FormatSpec("plain", torch.bfloat16, "token_latent", ("latent",), 4, (2, 4))
    manager = AttentionStateManager(2)
    store = MLAStateStore(adapter.state_specs(), fmt, manager)
    manager.finish_state_registration()
    (state,) = manager.get_full_attention_states()
    state.allocate_storage(8)
    assert state.get_token_bytes() == 2 * 576 * 2
    assert all(v.fields[0].shape == (12, 1, 576) for v in store.views().values())
    assert state.config.tp_rank == 2 and state.config.tp_world_size == 4


def test_flashinfer_mla_plans_have_separate_semantic_reservations(monkeypatch):
    monkeypatch.setattr("lightllm.common.execution.compute.attention.mla.flashinfer.find_spec", lambda _: object())
    fmt = FormatSpec("plain", torch.float16, "token_latent", ("latent",), 1)
    first = MLAAttentionSpec(32, 512, 128, 64, 128, torch.float16)
    second = replace(first, q_heads=128)
    backend = FlashInferMLABackend("decode")
    caps, limits = DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(512 * 1024 ** 2, 3, 1024, 2)
    size = backend.probe(first, fmt, caps, limits).offers[0].workspace_bytes
    planner = Planner((backend,), caps, replace(limits, workspace_bytes=size))
    requests = [
        AttentionRequest(
            i,
            "decode",
            spec,
            StateSpec("main", f"mla.{i}", i, (i,), MLAKVStateSpec(latent_dim=512, rope_dim=64)),
            fmt,
            MLAStaticParams(spec, torch.empty, None),
        )
        for i, spec in enumerate((first, second))
    ]
    with pytest.raises(ValueError, match="plan workspace"):
        planner.prepare(requests)
    requests[1] = replace(requests[1], spec=first, params=requests[0].params)
    assert planner.prepare(requests).workspace_bytes == size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("latent_dim", [256, 512])
@torch.no_grad()
def test_mla_verify_dynamic_groups_and_graph(implementation, dtype, latent_dim):
    from lightllm.common.execution.contracts.state import StateView

    torch.manual_seed(83)
    spec = MLAAttentionSpec(5, latent_dim, 128, 64, 128, dtype, 0.07)
    fmt = FormatSpec("plain", dtype, "token_latent", ("latent",), 4)
    cache = torch.randn(192, 1, latent_dim + 64, dtype=dtype, device="cuda") * 0.2
    view = StateView(
        StateSpec("main", "mla.0", 0, (0,), MLAKVStateSpec(latent_dim=latent_dim, rope_dim=64)), fmt, (cache,)
    )
    backend = (
        FlashInferMLABackend("verify")
        if implementation == "flashinfer"
        else MLAAttentionBackend(implementation, "verify")
    )
    allocate = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
    planner = Planner(
        (backend,), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(512 * 1024 ** 2, 6, 64, 2, max_verify_tokens=3)
    )
    bound = (
        planner.bind(
            [(0, "verify", spec, view, MLAStaticParams(spec, allocate, None), "auto")],
            lambda n: allocate((n,), torch.uint8),
        )
        .bindings[0]
        .execution
    )
    tensor = lambda data: torch.tensor(data, dtype=torch.int32, device="cuda")
    table = torch.randperm(192, device="cuda").int().view(3, 64)

    def metadata(rows, lengths, lane):
        return bound.create_metadata(
            AttentionBatch(
                6,
                1,
                max(lengths),
                lane,
                table,
                tensor(rows),
                tensor(lengths),
                tensor(range(6)),
                tensor(lengths) - 1,
                tensor(range(7)),
                tensor([0] + lengths).cumsum(0).int(),
                hold_request_id=2,
            )
        )

    def reference(q, rows, lengths):
        result = []
        for i, (row, length) in enumerate(zip(rows, lengths, strict=True)):
            kv = cache[table[row, :length], 0].float()
            scores = (q[i].float() @ kv.T) * spec.scale
            result.append(scores.softmax(-1) @ kv[:, :latent_dim])
        return torch.stack(result).to(dtype)

    rows, lengths = [0, 0, 0, 1, 1, 2], [13, 14, 15, 7, 8, 1]
    queries = [torch.randn(6, 5, latent_dim + 64, dtype=dtype, device="cuda") * 0.3 for _ in range(2)]
    metas = [metadata(rows, lengths, lane) for lane in range(2)]
    for q, meta in zip(queries, metas, strict=True):
        torch.testing.assert_close(
            bound.run((q[..., :latent_dim], q[..., latent_dim:]), meta),
            reference(q, rows, lengths),
            rtol=0.02,
            atol=0.004,
        )
    graph = torch.cuda.CUDAGraph()
    q = queries[0]
    with torch.cuda.graph(graph):
        captured = bound.run((q[..., :latent_dim], q[..., latent_dim:]), metas[0])
    # Change both the number of groups and the request order without recapturing.
    rows, lengths = [1, 1, 0, 0, 2, 2], [21, 22, 32, 33, 1, 1]
    q.mul_(0.7)
    metas[0].copy_for_graph(metadata(rows, lengths, 0))
    graph.replay()
    torch.testing.assert_close(captured, reference(q, rows, lengths), rtol=0.02, atol=0.004)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("latent_dim", [128, 256, 512])
@torch.no_grad()
def test_mla_prefix_graph_and_concurrent_lanes(
    implementation, phase, dtype, latent_dim, monkeypatch, q_heads=5, seq_capacity=1024
):
    torch.manual_seed(63)
    spec = MLAAttentionSpec(q_heads, latent_dim, 128, 64, 128, dtype, 0.07)
    fmt = FormatSpec("plain", dtype, "token_latent", ("latent",), 4)
    declaration = StateSpec("main", "mla.0", 0, (0,), MLAKVStateSpec(latent_dim=latent_dim, rope_dim=64))
    manager = AttentionStateManager(3)
    store = MLAStateStore((declaration,), fmt, manager)
    manager.finish_state_registration()
    (state,) = manager.get_full_attention_states()
    state.device = torch.device("cuda")
    state.allocate_storage(3 * seq_capacity)
    view = store.views()[declaration.key]
    latent = view.fields[0]
    latent.normal_(std=0.2)
    writer = MLAKVWriter(view)
    positions = torch.tensor([5, 1031], dtype=torch.int32, device="cuda")
    new = torch.randn(2, 1, latent_dim + 64, dtype=dtype, device="cuda") * 0.2
    writer(token_indices=positions, kv=new)
    torch.testing.assert_close(latent[positions], new, rtol=0, atol=0)
    projection = torch.randn(latent_dim, q_heads * 256, dtype=dtype, device="cuda") * 0.03
    params = MLAStaticParams(
        spec,
        lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        lambda x, out: torch.mm(x, projection, out=out),
    )
    planner = Planner(
        (
            FlashInferMLABackend(phase)
            if implementation == "flashinfer"
            else MLAAttentionBackend(implementation, phase),
        ),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(512 * 1024 * 1024, 3, seq_capacity, 2),
    )
    if implementation == "fa3" and phase == "decode" and latent_dim < 256:
        with pytest.raises(ValueError, match="latent width of at least 256"):
            planner.select(spec, fmt, phase)
        return
    bound = (
        planner.bind(
            [(0, phase, spec, view, params, "auto")],
            lambda size: torch.empty(size, dtype=torch.uint8, device="cuda"),
        )
        .bindings[0]
        .execution
    )
    monkeypatch.setattr(manager, "get_state", lambda *a, **k: pytest.fail("runtime resource lookup"))
    table = torch.randperm(3 * seq_capacity, device="cuda").to(torch.int32).view(3, seq_capacity)
    query_lengths = [3, 2, 1] if phase == "prefill" else [1, 1, 1]
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")

    def metadata(lengths, rows, lane):
        qcu = tensor([0] + query_lengths).cumsum(0).int()
        batch = AttentionBatch(
            3,
            max(query_lengths),
            max(lengths),
            lane,
            table,
            tensor(rows),
            tensor(lengths),
            qcu[:-1],
            tensor(lengths) - tensor(query_lengths),
            qcu,
            tensor([0] + lengths).cumsum(0).int(),
            total_kv_tokens=sum(lengths),
        )
        return bound.create_metadata(batch)

    def query():
        count = sum(query_lengths)
        if phase == "prefill":
            return torch.randn(count, q_heads, 192, device="cuda", dtype=dtype) * 0.3
        return (
            torch.randn(3, q_heads, latent_dim, device="cuda", dtype=dtype) * 0.3,
            torch.randn(3, q_heads, 64, device="cuda", dtype=dtype) * 0.3,
        )

    def reference(inputs, lengths, rows):
        outputs, start = [], 0
        for length, row, count in zip(lengths, rows, query_lengths):
            cache = latent[table[row, :length].long(), 0]
            if phase == "prefill":
                expanded = (cache[:, :latent_dim] @ projection).reshape(length, q_heads, 256)
                key = torch.cat((expanded[..., :128], cache[:, None, latent_dim:].expand(-1, q_heads, -1)), dim=-1)
                value = expanded[..., 128:]
                q = inputs[start : start + count]
            else:
                key = cache[:, None].expand(-1, q_heads, -1)
                value = key[..., :latent_dim]
                q = torch.cat((inputs[0][start : start + 1], inputs[1][start : start + 1]), dim=-1)
            scores = torch.einsum("qhd,khd->hqk", q.float(), key.float()) * spec.scale
            mask = (
                torch.arange(length, device="cuda")[None]
                > (length - count + torch.arange(count, device="cuda"))[:, None]
            )
            scores.masked_fill_(mask, -torch.inf)
            output = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value.float())
            outputs.append(output.to(dtype))
            start += count
        return torch.cat(outputs)

    lengths, rows = [513, 257, 9], [0, 2, 1]
    queries = [query(), query()]
    metas = [metadata(lengths, rows, lane) for lane in range(2)]
    for q, meta in zip(queries, metas):
        torch.testing.assert_close(bound.run(q, meta), reference(q, lengths, rows), rtol=0.02, atol=0.004)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream, q, meta in zip(streams, queries, metas):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            outputs.append(bound.run(q, meta))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for q, output in zip(queries, outputs):
        torch.testing.assert_close(output, reference(q, lengths, rows), rtol=0.02, atol=0.004)
    if phase == "decode":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound.run(queries[0], metas[0])
        lengths, rows = [seq_capacity - 47, seq_capacity - 243, 67], [2, 1, 0]
        metas[0].copy_for_graph(metadata(lengths, rows, 0))
        graph.replay()
        torch.testing.assert_close(captured, reference(queries[0], lengths, rows), rtol=0.02, atol=0.004)
        if implementation == "flashinfer":
            lengths, rows = [5, 257, 513], [1, 0, 2]
            metas[0].copy_for_graph(metadata(lengths, rows, 0))
            graph.replay()
            torch.testing.assert_close(captured, reference(queries[0], lengths, rows), rtol=0.02, atol=0.004)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flashinfer_mla_long_multi_cluster_graph(monkeypatch):
    test_mla_prefix_graph_and_concurrent_lanes(
        "flashinfer", "decode", torch.bfloat16, 512, monkeypatch, q_heads=128, seq_capacity=4096
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_flashinfer_mla_distinct_layers_preserve_plans():
    from lightllm.common.execution.contracts.state import StateView

    torch.manual_seed(67)
    dtype = torch.bfloat16
    fmt = FormatSpec("plain", dtype, "token_latent", ("latent",), 1)
    cache = torch.randn(128, 1, 576, dtype=dtype, device="cuda") * 0.1
    specs = [MLAAttentionSpec(h, 512, 128, 64, 128, dtype, s) for h, s in ((32, 0.05), (128, 0.1))]
    allocate = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
    requests = []
    for i, spec in enumerate(specs):
        view = StateView(
            StateSpec("main", f"mla.{i}", i, (i,), MLAKVStateSpec(latent_dim=512, rope_dim=64)), fmt, (cache.clone(),)
        )
        requests.append((i, "decode", spec, view, MLAStaticParams(spec, allocate, None), "auto"))
    planner = Planner(
        (FlashInferMLABackend("decode"),), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(512 * 1024 ** 2, 2, 64)
    )
    bound = [b.execution for b in planner.bind(requests, lambda n: allocate((n,), torch.uint8)).bindings]
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    table = torch.arange(128, dtype=torch.int32, device="cuda").view(2, 64)
    batch = AttentionBatch(
        2,
        1,
        31,
        0,
        table,
        tensor([0, 1]),
        tensor([31, 17]),
        tensor([0, 1]),
        tensor([30, 16]),
        tensor([0, 1, 2]),
        tensor([0, 31, 48]),
        total_kv_tokens=48,
    )
    # Preparing every layer before running the first must not overwrite its integer plan.
    metas = [b.create_metadata(batch) for b in bound]
    assert metas[0].wrapper._int_workspace_buffer.data_ptr() != metas[1].wrapper._int_workspace_buffer.data_ptr()
    for execution, meta in zip(bound, metas):
        q = torch.randn(2, execution.spec.q_heads, 576, dtype=dtype, device="cuda") * 0.2
        result = execution.run((q[..., :512], q[..., 512:]), meta)
        expected = []
        for row, length in enumerate((31, 17)):
            kv = cache[table[row, :length], 0].float()
            scores = (q[row].float() @ kv.T) * execution.spec.scale
            expected.append((scores.softmax(-1) @ kv[:, :512]).to(dtype))
        torch.testing.assert_close(result, torch.stack(expected), rtol=0.02, atol=0.004)
