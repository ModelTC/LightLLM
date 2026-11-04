from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.execution import (
    DeviceCaps,
    ExecutionLimits,
    FormatSpec,
    Planner,
    StateSpec,
    SparseMLAAttentionSpec,
)
from lightllm.common.execution.dense import DenseBatchMetadata
from lightllm.common.execution.sparse_mla import (
    PackedMLAKVWriter,
    SparseMLAAttentionBackend,
    SparseMLAStaticParams,
    SparseMLAInputs,
)
from lightllm.common.execution.state_store import PackedMLAStateStore


def format_spec(quantized, shard=(0, 1)):
    dtype = torch.uint8 if quantized else torch.bfloat16
    return FormatSpec(
        "fp8_group128" if quantized else "plain",
        dtype,
        "token_packed_mla",
        ("latent", "indexer"),
        4,
        shard,
        group_size=128 if quantized else None,
        scale_dtype=torch.float32 if quantized else None,
        field_dtypes=(dtype, torch.uint8),
    )


def test_sparse_model_declares_indexer_and_tp_replicated_storage():
    from lightllm.models.deepseek3_2.model import Deepseek3_2ModelAdapter

    adapter = Deepseek3_2ModelAdapter(
        dict(
            num_hidden_layers=2,
            num_attention_heads=128,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=2048,
        ),
        role="main",
        dtype=torch.bfloat16,
        tp_world_size=8,
    )
    spec = adapter.attention_specs()[0]
    assert isinstance(spec, SparseMLAAttentionSpec) and spec.q_heads == 16 and spec.index_heads == 64
    for quantized in (False, True):
        manager = AttentionStateManager(2)
        store = PackedMLAStateStore(adapter.state_specs(), format_spec(quantized, (3, 8)), manager)
        manager.finish_state_registration()
        state = manager.get_full_attention_states()[0]
        state.allocate_storage(12)
        assert state.config.tp_rank == 3 and state.config.tp_world_size == 8
        assert all(v.fields[0].shape == (16, 1, 656 if quantized else 576) for v in store.views().values())


def test_flashmla_rejects_unsupported_native_layout_and_kernel_alignment():
    spec = SparseMLAAttentionSpec(64, 512, 128, 64, 128, torch.bfloat16, index_topk=256)
    caps, limits = DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(10 ** 7, 2, 512)
    backend = SparseMLAAttentionBackend("flashmla_sparse", "prefill")
    assert backend.probe(spec, format_spec(False), caps, limits).offers
    for other, fmt in (
        (replace(spec, q_heads=16), format_spec(False)),
        (replace(spec, index_topk=32), format_spec(False)),
    ):
        assert not backend.probe(other, fmt, caps, limits).offers


def test_fp8_flashmla_checks_library_api_before_binding(monkeypatch):
    from types import SimpleNamespace
    from lightllm.common.execution import sparse_mla

    spec = SparseMLAAttentionSpec(16, 512, 128, 64, 128, torch.bfloat16, index_topk=256)
    caps, limits = DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(10 ** 7, 2, 512)
    backend = SparseMLAAttentionBackend("flashmla_sparse", "decode")
    monkeypatch.setattr(sparse_mla, "find_spec", lambda name: None if name == "flash_mla" else object())
    assert "standalone" in backend.probe(spec, format_spec(True), caps, limits).reason
    monkeypatch.setattr(sparse_mla, "find_spec", lambda name: object())
    library = SimpleNamespace(get_mla_metadata=lambda required_lengths: None)
    monkeypatch.setattr(sparse_mla, "import_module", lambda name: library)
    assert "incompatible" in backend.probe(spec, format_spec(True), caps, limits).reason
    library.get_mla_metadata = lambda: (object(), None)
    library.flash_mla_with_kvcache = lambda indices, is_fp8_kvcache, tile_scheduler_metadata: None
    library.flash_mla_sparse_fwd = lambda: None
    assert backend.probe(spec, format_spec(True), caps, limits).offers
    for invalid in (replace(spec, q_heads=129), replace(spec, index_topk=32)):
        assert not backend.probe(invalid, format_spec(True), caps, limits).offers


def create_bound(
    quantized, phase, implementation="triton", heads=5, *, max_seq=512, topk=256, index_heads=64, scale_fmt=None
):
    spec = SparseMLAAttentionSpec(
        heads,
        512,
        128,
        64,
        128,
        torch.bfloat16,
        0.07,
        index_topk=topk,
        index_heads=index_heads,
        index_scale_fmt=scale_fmt,
    )
    fmt = format_spec(quantized)
    declaration = StateSpec("main", "sparse.0", 0, (0,), (512, 64, 128))
    manager = AttentionStateManager(2)
    store = PackedMLAStateStore((declaration,), fmt, manager)
    manager.finish_state_registration()
    state = manager.get_full_attention_states()[0]
    state.device = torch.device("cuda")
    state.allocate_storage(max_seq * 2)
    view = store.views()[declaration.key]
    alloc = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
    planner = Planner(
        (SparseMLAAttentionBackend(implementation, phase),),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(10 ** 7, 2, max_seq, 2),
    )
    bound = (
        planner.bind(
            [(0, phase, spec, view, SparseMLAStaticParams(spec, alloc), "auto")],
            lambda size: alloc((size,), torch.uint8),
        )
        .bindings[0]
        .execution
    )
    return bound, view, manager


def metadata(bound, table, lengths, rows, counts, lane=0):
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
    cuq = tensor([0] + counts).cumsum(0).int()
    batch = DenseBatchMetadata(
        2,
        max(counts),
        max(lengths),
        lane,
        table,
        tensor(rows),
        tensor(lengths),
        cuq[:-1],
        tensor(lengths) - tensor(counts),
        cuq,
        tensor([0] + lengths).cumsum(0).int(),
        total_kv_tokens=sum(lengths),
        total_q_tokens=sum(counts),
    )
    return bound.create_metadata(batch)


def decoded_cache(view):
    kv = view.fields[0]
    if kv.dtype == torch.bfloat16:
        return kv
    values = kv[..., :512].view(torch.float8_e4m3fn).float()
    scale = kv[..., 512:528].view(torch.float32).repeat_interleave(128, -1)
    return torch.cat(((values * scale).bfloat16(), kv[..., 528:656].view(torch.bfloat16)), -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@torch.no_grad()
def test_sparse_attention_matches_torch_with_prefix_and_graph(
    quantized, phase, monkeypatch, implementation="triton", heads=None
):
    torch.manual_seed(137)
    bound, view, manager = create_bound(
        quantized, phase, implementation, heads or (64 if implementation == "flashmla_sparse" else 5)
    )
    view.fields[1].fill_(173)
    original = torch.randn((1028, 1, 576), dtype=torch.bfloat16, device="cuda") * 0.2
    original[0].zero_()
    PackedMLAKVWriter(view)(token_indices=torch.arange(1028, dtype=torch.int32, device="cuda"), kv=original)
    assert bool((view.fields[1] == 173).all())
    if quantized:
        grouped = original[..., :512].float().view(1028, 4, 128)
        scales = torch.exp2(torch.ceil(torch.log2((grouped.abs().amax(-1) / 448).clamp_min(1e-4))))
        expected = (grouped / scales[..., None]).to(torch.float8_e4m3fn).view(1028, 1, 512)
        torch.testing.assert_close(view.fields[0][..., :512], expected.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(view.fields[0][..., 512:528].view(torch.float32)[:, 0], scales, rtol=0, atol=0)
    cache = decoded_cache(view)
    monkeypatch.setattr(manager, "get_state", lambda **_: pytest.fail("runtime resource lookup"))
    table = torch.randperm(1024, device="cuda").int().view(2, 512)
    counts = [3, 2] if phase == "prefill" else [1, 1]
    query = torch.randn(sum(counts), bound.spec.q_heads, 576, device="cuda", dtype=torch.bfloat16) * 0.3

    def scenario(lengths, rows):
        physical, logical, current, outputs = [], [], [], []
        q_start, k_start = 0, 0
        for length, row, count in zip(lengths, rows, counts):
            ready = length - count
            current.append(original[table[row, ready:length].long()])
            for i in range(count):
                valid = ready + i + 1
                selected = torch.randperm(valid, device="cuda")[:256]
                mem = table[row, selected.long()]
                keys = cache[mem.long(), 0].clone()
                if phase == "prefill" and quantized:
                    fresh = selected >= ready
                    keys[fresh] = original[mem[fresh].long(), 0]
                probabilities = (query[q_start + i].float() @ keys.float().T * bound.spec.scale).softmax(-1)
                outputs.append(probabilities @ keys[:, :512].float())
                physical.append(torch.nn.functional.pad(mem, (0, 256 - len(mem)), value=-1))
                logical.append(torch.nn.functional.pad((selected + k_start).int(), (0, 256 - len(mem)), value=-1))
            q_start += count
            k_start += length
        return (
            torch.stack(physical).int(),
            torch.stack(logical).int(),
            torch.cat(current),
            torch.stack(outputs).bfloat16(),
        )

    first_lengths, first_rows = [301, 19], [0, 1]
    meta = metadata(bound, table, first_lengths, first_rows, counts)
    physical, logical, current, expected = scenario(first_lengths, first_rows)
    result = bound.attend(query, physical, logical, meta, current)
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.003)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            bound.attend(query, physical, logical, meta, current)
    torch.cuda.current_stream().wait_stream(stream)
    if bound.requires_dynamic_prefill:
        from lightllm.common.execution.graph_program import GraphProgram

        graph = GraphProgram(torch.cuda.graph_pool_handle())
        with graph.capture():
            replay = graph.call(
                lambda tensors, batch: (bound.attend(*tensors[:3], batch, tensors[3]),),
                (query, physical, logical, current),
                meta,
                (((query.shape[0], bound.spec.q_heads, 512), query.dtype, query.device),),
            )[0]
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay = bound.attend(query, physical, logical, meta, current)
    for lengths, rows in (([419, 279], [1, 0]), ([7, 9], [0, 1])):
        updated = metadata(bound, table, lengths, rows, counts)
        p, l, c, expected = scenario(lengths, rows)
        physical.copy_(p)
        logical.copy_(l)
        current.copy_(c)
        meta.copy_for_graph(updated)
        graph.replay(meta) if bound.requires_dynamic_prefill else graph.replay()
        torch.testing.assert_close(replay, expected, rtol=0.02, atol=0.003)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_flashmla_native_packed_attention(monkeypatch, phase):
    test_sparse_attention_matches_torch_with_prefix_and_graph(False, phase, monkeypatch, "flashmla_sparse")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("heads", [16, 64, 128])
def test_flashmla_fp8_native_packed_attention(monkeypatch, phase, heads):
    test_sparse_attention_matches_torch_with_prefix_and_graph(True, phase, monkeypatch, "flashmla_sparse", heads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "max_seq,topk,index_heads,scale_fmt", [(512, 256, 64, None), (4096, 2048, 64, None), (512, 256, 6, "ue8m0")]
)
@torch.no_grad()
def test_sparse_indexer_selection_and_decode_graph(
    monkeypatch, max_seq, topk, index_heads, scale_fmt, implementation="triton"
):
    from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant
    from lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks import destindex_copy_indexer_ks

    torch.manual_seed(89)
    bound, view, manager = create_bound(
        True, "decode", implementation, max_seq=max_seq, topk=topk, index_heads=index_heads, scale_fmt=scale_fmt
    )
    monkeypatch.setattr(manager, "get_state", lambda **_: pytest.fail("runtime resource lookup"))
    capacity = view.fields[0].shape[0]
    latent = torch.randn(capacity, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.2
    PackedMLAKVWriter(view)(token_indices=torch.arange(capacity, device="cuda", dtype=torch.int32), kv=latent)
    history = torch.randn(capacity, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    kfp8, kscale = act_quant(history, scale_fmt=scale_fmt)
    destindex_copy_indexer_ks(kfp8, kscale, torch.arange(capacity, device="cuda", dtype=torch.int32), view.fields[1])
    table = torch.randperm(max_seq * 2, device="cuda").int().view(2, max_seq)
    lengths = [max_seq - 141, max_seq - 213]
    meta = metadata(bound, table, lengths, [0, 1], [1, 1])
    iq = torch.randn(2, index_heads, 128, device="cuda", dtype=torch.bfloat16) * 0.3
    weights = torch.randn(2, index_heads, device="cuda", dtype=torch.bfloat16) * 0.1
    query = torch.randn(2, 5, 576, device="cuda", dtype=torch.bfloat16) * 0.3
    inputs = SparseMLAInputs(query, iq, history[table[:, 0].long()], weights, table[:, 0])
    original_logits = bound.index_logits
    captured = []

    def record_logits(*args, **kwargs):
        result = original_logits(*args, **kwargs)
        captured.append(result.clone())
        return result

    monkeypatch.setattr(bound, "index_logits", record_logits)
    physical, logical = bound.select_indices(inputs, meta)
    monkeypatch.setattr(bound, "index_logits", original_logits)
    qfp8, qscale = act_quant(iq, scale_fmt=scale_fmt)
    for i, length in enumerate(lengths):
        mem = table[i, :length].long()
        scores = torch.relu(qfp8[i].float() @ kfp8[mem].float().T)
        scores = (scores * (weights[i].float() * qscale[i, :, 0])[:, None]).sum(0) * kscale[mem, 0]
        selected = torch.topk(scores, topk).indices
        torch.testing.assert_close(captured[0][i, :length], scores, rtol=1e-4, atol=1e-5)
        assert torch.equal(physical[i].sort().values, table[i, selected].sort().values)
    expected = bound.attend(query, physical, logical, meta)

    def assert_output(result, expected):
        if topk != 2048:
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            return
        # SGL top-k may permute the same selected tokens between launches.
        error = result.float() - expected.float()
        assert error.square().mean().sqrt() / expected.float().square().mean().sqrt() < 0.01
        assert error.abs().max() < expected.abs().max() / 128

    assert_output(bound.run(inputs, meta), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            bound.run(inputs, meta)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replay_physical, replay_logical = bound.select_indices(inputs, meta)
        replay = bound.attend(query, replay_physical, replay_logical, meta)
    for lengths, rows in (([max_seq - 33, max_seq - 103], [1, 0]), ([17, 11], [0, 1])):
        updated = metadata(bound, table, lengths, rows, [1, 1])
        physical, logical = bound.select_indices(inputs, updated)
        expected = bound.attend(query, physical, logical, updated)
        meta.copy_for_graph(updated)
        graph.replay()
        assert torch.equal(physical.sort().values, replay_physical.sort().values)
        assert_output(replay, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flashmla_fp8_indexer_long_decode_graph(monkeypatch):
    test_sparse_indexer_selection_and_decode_graph(monkeypatch, 4096, 2048, 64, None, "flashmla_sparse")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_flashmla_fp8_scheduler_groups_and_lanes():
    from types import SimpleNamespace
    from lightllm.common.execution.runner import Runner
    from lightllm.common.execution.device_runtime import CudaRuntime

    specs = {
        i: SparseMLAAttentionSpec(heads, 512, 128, 64, 128, torch.bfloat16, index_topk=256)
        for i, heads in enumerate((16, 64, 128))
    }
    declarations = tuple(StateSpec("main", f"sparse.{i}", i, (i,), (512, 64, 128)) for i in specs)
    adapter = SimpleNamespace(attention_specs=lambda: specs, state_specs=lambda: declarations)
    layers = [SimpleNamespace(layer_num_=i) for i in specs]
    manager, runtime = AttentionStateManager(2), CudaRuntime()
    runner = Runner(
        adapter,
        layers,
        manager,
        runtime,
        ExecutionLimits(10 ** 7, 2, 512, 2),
        format_spec(True),
        {"prefill": "triton", "decode": "flashmla_sparse"},
        static_params={i: SparseMLAStaticParams(spec, runtime.allocate_tensor) for i, spec in specs.items()},
    )
    manager.finish_state_registration()
    for state in manager.get_full_attention_states():
        state.device = torch.device("cuda")
        state.allocate_storage(1024)
    runner.bind(layers)
    bounds = [layer.bound_decode_attention for layer in layers]
    assert sum(source.phase == "decode" for source in runner.metadata_sources.values()) == 2
    table = torch.arange(1024, device="cuda", dtype=torch.int32).view(2, 512)

    def prepare(lengths, rows, lane=0):
        tensor = lambda data: torch.tensor(data, device="cuda", dtype=torch.int32)
        batch = SimpleNamespace(
            is_prefill=False,
            batch_size=2,
            max_q_seq_len=1,
            max_kv_seq_len=max(lengths),
            microbatch_index=lane,
            req_manager=SimpleNamespace(req_to_token_indexs=table),
            b_req_idx=tensor(rows),
            b_seq_len=tensor(lengths),
            b_q_start_loc=tensor([0, 1]),
            b_ready_cache_len=tensor(lengths) - 1,
            b1_cu_q_seq_len=tensor([0, 1, 2]),
            b1_cu_kv_seq_len=tensor([0, lengths[0], sum(lengths)]),
            total_token_num=sum(lengths),
            input_ids=tensor([1, 2]),
        )
        batch.execution_metadata = runner.prepare_batch(batch)
        return batch

    batch = prepare([8, 7], [0, 1])
    first = bounds[0].prepare_batch(batch)
    assert bounds[1].prepare_batch(batch) is first
    assert bounds[2].prepare_batch(batch).scheduler is not first.scheduler
    other = bounds[0].prepare_batch(prepare([12, 13], [1, 0], lane=1))
    assert first.scheduler is not other.scheduler
    assert first.ragged.data_ptr() != other.ragged.data_ptr()
    for bound in bounds:
        bound.cache.zero_()
        bound.scales.fill_(1)
        query = torch.zeros((2, bound.spec.q_heads, 576), device="cuda", dtype=torch.bfloat16)
        indices = torch.full((2, 256), -1, device="cuda", dtype=torch.int32)
        indices[:, 0] = 0
        meta = bound.prepare_batch(batch)
        output = bound.attend(query, indices, indices, meta)
        assert torch.equal(output, torch.zeros_like(output))
        scheduler = meta.scheduler
        meta.copy_for_graph(metadata(bound, table, [10, 11], [1, 0], [1, 1]))
        assert meta.scheduler is scheduler
