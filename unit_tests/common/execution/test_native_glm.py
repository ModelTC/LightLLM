"""Pooled history/window contracts and independent KDA/NoPE numerical checks."""

from lightllm.common.execution.formats.recurrent.spec import ConvSSMStateSpec
from lightllm.common.execution.formats.kv.spec import IndexerWindowStateSpec
from lightllm.common.execution.formats.kv.spec import PooledKVStateSpec


from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.execution.compute.attention.batch import AttentionBatch
from lightllm.common.execution.compute.attention.kda.backend import KDAInputs, KDAStaticParams, KimiDeltaAttentionBackend
from lightllm.common.execution.planner.planner import AttentionRequest, Planner
from lightllm.common.execution.compute.attention.pooled_nope.backend import (
    PooledNoPEAttentionBackend,
    PooledNoPEInputs,
    PooledNoPEStaticParams,
)
from lightllm.common.execution.formats.kv.pooled_nope import PooledNoPEKVWriter
from lightllm.common.execution.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.common.execution.contracts.state import FormatSpec, StateSpec, StateView
from lightllm.common.execution.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.common.execution.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec
from lightllm.common.execution.storage.attention import ConvSSMStateStore, IndexerWindowStateStore, PackedMLAStateStore


def pooled_contracts(quantized=False, dtype=torch.bfloat16):
    spec = PooledNoPEAttentionSpec(4, 512, 128 ** -0.5, 4, 128, 256, 4, dtype)
    state = StateSpec("main", "pooled_kv.3", 3, (3,), PooledKVStateSpec(latent_dim=512, index_dim=128))
    tail = StateSpec(
        "main", "indexer_tail.3", 3, (3,), IndexerWindowStateSpec(window=4, stride=4, index_dim=128), "window"
    )
    stored = torch.uint8 if quantized else dtype
    fmt = FormatSpec(
        "fp8_group64_e8m0" if quantized else "plain",
        stored,
        "token_pooled_nope",
        ("latent", "indexer"),
        1,
        group_size=64 if quantized else None,
        scale_dtype=torch.uint8 if quantized else None,
        field_dtypes=(stored, torch.uint8),
    )
    window = FormatSpec("plain", dtype, "request_indexer_window", ("tail",), 1)
    return spec, state, tail, fmt, window


def make_batch(lengths, rows, ready, table=None, lane=0, device="cuda"):
    tensor = lambda v: torch.tensor(v, device=device, dtype=torch.int32)
    seq, cuq, cuk = [a + b for a, b in zip(lengths, ready)], [0], [0]
    for q, k in zip(lengths, seq):
        cuq.append(cuq[-1] + q)
        cuk.append(cuk[-1] + k)
    return AttentionBatch(
        len(rows),
        max(lengths),
        max(seq),
        lane,
        table,
        tensor(rows),
        tensor(seq),
        tensor(cuq[:-1]),
        tensor(ready),
        tensor(cuq),
        tensor(cuk),
        total_q_tokens=sum(lengths),
        total_kv_tokens=sum(seq),
    )


def test_pooled_plan_requires_independent_compatible_tail():
    spec, state, tail, fmt, window = pooled_contracts()
    params = PooledNoPEStaticParams(spec, torch.empty, torch.empty(4, 128))
    planner = Planner(
        (PooledNoPEAttentionBackend("triton", "prefill"),), DeviceCaps("cuda"), ExecutionLimits(8192, 2, 64)
    )
    request = AttentionRequest(
        3, "prefill", spec, state, fmt, params, auxiliary_states=(("indexer_tail", tail, window),)
    )
    assert planner.prepare([request]).workspace_bytes == 512
    prepared = planner.prepare([request])
    with pytest.raises(ValueError, match="allocated auxiliary state"):
        prepared.bind(
            {
                state.key: StateView(state, fmt, (None, None)),
                tail.key: StateView(tail, replace(window, dtype=torch.float16), (None,)),
            },
            {},
        )
    for bad, message in [
        ((), "separately declared"),
        ((("indexer_tail", replace(tail, retention="history"), window),), "tail window"),
        ((("indexer_tail", replace(tail, role="draft0"), window),), "same model role"),
        ((("indexer_tail", tail, replace(window, dtype=torch.float16)),), "compatible independent"),
    ]:
        with pytest.raises(ValueError, match=message):
            planner.prepare([replace(request, auxiliary_states=bad)])


def test_pooled_sgl_prefill_rejects_fp16_before_kernel_execution():
    spec, _, _, fmt, window = pooled_contracts(dtype=torch.float16)
    result = PooledNoPEAttentionBackend("flashmla_sparse", "prefill").probe(
        spec, fmt, DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(8192, 2, 64), {"indexer_tail": window}
    )
    assert not result.offers
    assert "BF16" in result.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_kda_fresh_prefix_checkpoint_and_graph(dtype, monkeypatch):
    dim = 128
    torch.manual_seed(117)
    spec = KimiDeltaAttentionSpec(2, dim, 4, dtype)
    state = StateSpec(
        "main",
        "kda.0",
        0,
        (0,),
        ConvSSMStateSpec(key_heads=2, value_heads=2, key_dim=dim, value_dim=dim, conv_history_length=3),
        "recurrent",
    )
    fmt = FormatSpec("conv_ssm", dtype, "request_conv_ssm", ("conv", "ssm"), 1, field_dtypes=(dtype, torch.float32))
    manager = AttentionStateManager(4)
    store = ConvSSMStateStore((state,), fmt, manager, 4, "cuda", kind="kda")
    view = store.views()[state.key]
    conv, ssm = view.fields
    conv.normal_(std=0.1)
    ssm.normal_(std=0.05)
    params = KDAStaticParams(
        spec,
        torch.randn(6 * dim, 4, device="cuda", dtype=dtype) * 0.2,
        torch.randn(2, device="cuda") * 0.1,
        torch.randn(2 * dim, device="cuda") * 0.1,
    )
    planner = Planner(
        tuple(KimiDeltaAttentionBackend(p) for p in ("prefill", "decode")),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(0, 4, 256, 2),
    )
    plan = planner.bind(
        [(0, p, spec, view, params, "auto") for p in ("prefill", "decode")], lambda _: pytest.fail("scratch")
    )
    prefill, decode = [b.execution for b in plan.bindings]
    monkeypatch.setattr(manager, "get_state", lambda *a, **k: pytest.fail("runtime manager lookup"))
    expected_conv, expected_ssm = conv.clone(), ssm.clone()

    def inputs(count):
        return KDAInputs(
            torch.randn(count, 6 * dim, device="cuda", dtype=dtype) * 0.3,
            torch.randn(count, 2 * dim, device="cuda", dtype=dtype) * 0.3,
            torch.randn(count, 2, device="cuda", dtype=dtype) * 0.3,
        )

    def reference(data, lengths, rows, ready):
        outputs, offset = [], 0
        for row, length, prefix in zip(rows, lengths, ready):
            history = (
                expected_conv[row].float() if prefix else torch.zeros_like(expected_conv[row], dtype=torch.float32)
            )
            memory = expected_ssm[row].float() if prefix else torch.zeros_like(expected_ssm[row])
            for t in range(offset, offset + length):
                window = torch.cat((history, data.qkv[t].float()[:, None]), 1)
                mixed = F.silu((window * params.conv_weight.float()).sum(-1)).to(dtype).float()
                history = window[:, 1:]
                q, k, v = (x.reshape(2, dim) for x in mixed.split(2 * dim))
                q = F.normalize(q, p=2, dim=-1, eps=1e-6)
                k = F.normalize(k, p=2, dim=-1, eps=1e-6)
                decay = (
                    spec.gate_lower_bound
                    * torch.sigmoid(
                        params.a_log.exp()[:, None] * (data.gate[t].float().view(2, dim) + params.dt_bias.view(2, dim))
                    )
                ).exp()
                memory = memory * decay[:, :, None]
                delta = (v - torch.einsum("hkv,hk->hv", memory, k)) * data.beta[t].float().sigmoid()[:, None]
                memory = memory + k[:, :, None] * delta[:, None, :]
                outputs.append(torch.einsum("hkv,hk->hv", memory, q) * dim ** -0.5)
            expected_conv[row].copy_(history)
            expected_ssm[row].copy_(memory)
            offset += length
        return torch.stack(outputs)

    def check(actual, expected):
        torch.testing.assert_close(actual.reshape(-1, 2, dim).float(), expected, rtol=0.03, atol=0.003)
        torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(ssm, expected_ssm, rtol=0.03, atol=0.004)

    rows, lengths, ready = [2, 0], [67, 35], [5, 0]
    data = inputs(sum(lengths))
    target = reference(data, lengths, rows, ready)
    check(prefill.run(data, prefill.create_metadata(make_batch(lengths, rows, ready))), target)
    native = store.resources[state.key][1]
    native.initialize_page_pools(2, device="cuda", pin_memory=False)
    native.save_bounded_att_runtime_to_pages(rows, [72, 35], "big", [0, 1])
    conv[rows] = 0
    ssm[rows] = 0
    native.restore_bounded_att_runtime_from_pages(rows, [72, 35], "big", [0, 1])
    data = inputs(2)
    meta = decode.create_metadata(make_batch([1, 1], rows, [72, 35]))
    target = reference(data, [1, 1], rows, [72, 35])
    check(decode.run(data, meta), target)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode.run(data, meta)
    rows = [0, 2]
    meta.copy_for_graph(decode.create_metadata(make_batch([1, 1], rows, [37, 74])))
    data.qkv.add_(0.01)
    target = reference(data, [1, 1], rows, [37, 74])
    graph.replay()
    check(output, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,quantized,dtype",
    [
        ("triton", False, torch.bfloat16),
        ("triton", True, torch.bfloat16),
        ("flashmla_sparse", False, torch.bfloat16),
        ("flashmla_sparse", True, torch.bfloat16),
        ("triton", False, torch.float16),
        ("flashmla_sparse", False, torch.float16),
    ],
)
@torch.no_grad()
def test_pooled_nope_dense_and_sparse_prefix_graph(
    implementation, quantized, dtype, monkeypatch, verify_candidates=False
):
    torch.manual_seed(131)
    spec, state, tail, fmt, window = pooled_contracts(quantized, dtype)
    if verify_candidates:
        window = replace(window, candidate_slots=3)
    phases = ("prefill", "verify" if verify_candidates else "decode")
    manager = AttentionStateManager(4)
    store = PackedMLAStateStore((state,), fmt, manager)
    windows = IndexerWindowStateStore((tail,), window, manager, 4, "cuda")
    manager.finish_state_registration()
    raw_state = store.resources[state.key][1]
    raw_state.device = torch.device("cuda")
    raw_state.allocate_storage(2048)
    views = {**store.views(), **windows.views()}
    params = PooledNoPEStaticParams(
        spec,
        lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        torch.randn(4, 128, device="cuda", dtype=torch.float32) * 0.1,
    )
    planner = Planner(
        tuple(
            PooledNoPEAttentionBackend(backend, p)
            for backend in dict.fromkeys((implementation, "triton"))
            for p in phases
        ),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(
            1 << 20, 6 if verify_candidates else 4, 512, 2, max_verify_tokens=3 if verify_candidates else 1
        ),
    )
    prepared = planner.prepare(
        [
            AttentionRequest(3, p, spec, state, fmt, params, auxiliary_states=(("indexer_tail", tail, window),))
            for p in phases
        ]
    )
    plan = prepared.bind(views, prepared.reserve(lambda n: torch.empty(n, dtype=torch.uint8, device="cuda")))
    prefill, decode = [b.execution for b in plan.bindings]
    assert prefill.implementation == ("triton" if dtype == torch.float16 else implementation)
    assert decode.implementation == implementation
    writer = PooledNoPEKVWriter(views[state.key])
    monkeypatch.setattr(manager, "get_state", lambda *a, **k: pytest.fail("runtime manager lookup"))
    table = torch.randperm(2048, device="cuda", dtype=torch.int32).view(4, 512)
    history = torch.randn(2048, 1, 512, device="cuda", dtype=spec.dtype) * 0.2
    writer(token_indices=torch.arange(2048, device="cuda", dtype=torch.int32), kv=history)

    def data(count):
        return PooledNoPEInputs(
            torch.randn(count, 4, 512, device="cuda", dtype=spec.dtype) * 0.2,
            torch.randn(count, 128, device="cuda", dtype=spec.dtype) * 0.3,
            torch.randn(count, 128, device="cuda", dtype=spec.dtype) * 0.3,
            torch.randn(count, 4, 128, device="cuda", dtype=spec.dtype) * 0.3,
            torch.randn(count, 4, device="cuda"),
        )

    def expected(inputs, meta, physical):
        cache = prefill.cache
        if quantized:
            payload = cache[..., :448].view(torch.float8_e4m3fn).float()
            scale = torch.exp2(cache[..., 576:583].float() - 127).repeat_interleave(64, -1)
            cache = torch.cat((payload * scale, cache[..., 448:576].view(torch.bfloat16).float()), -1).to(spec.dtype)
        outputs = []
        for row, indices in enumerate(physical):
            idx = indices[indices >= 0].long()
            kv = cache[idx, 0].float()
            if quantized and inputs.current_kv is not None:
                # Current prefill tokens retain their unquantized representation.
                for j, token in enumerate(idx):
                    matches = (meta.ragged[meta.fresh_start[row] : meta.ke[row]] == token).nonzero().flatten()
                    if matches.numel():
                        kv[j] = inputs.current_kv[
                            meta.fresh_start[row] + meta.fresh_offset[row] + matches[0], 0
                        ].float()
            scores = inputs.query[row].float() @ kv.T * spec.score_scale
            outputs.append(scores.softmax(-1) @ kv if idx.numel() else torch.zeros_like(inputs.query[row]).float())
        return torch.stack(outputs)

    # Short prefix closes pools before a continuation exercises the sparse branch.
    rows, lengths, ready = [2, 0], [259, 267], [0, 0]
    inputs = data(sum(lengths))
    locations = torch.cat([table[r, :n] for r, n in zip(rows, lengths)])
    fresh = torch.randn(len(locations), 1, 512, device="cuda", dtype=spec.dtype) * 0.2
    history[locations.long()] = fresh
    writer(token_indices=locations, kv=fresh)
    inputs = replace(inputs, current_kv=fresh)
    meta = prefill.create_metadata(make_batch(lengths, rows, ready, table))
    physical, _ = prefill.select_indices(inputs, meta)
    output = prefill.run(inputs, meta)
    torch.testing.assert_close(output.float(), expected(inputs, meta, physical), rtol=0.02, atol=0.002)
    # Same pooled scores must select the independent FP32 weighted-ReLU top-k set.
    from lightllm.common.execution.formats.kv.kernels.index_quant import hadamard_transform_quant_fp8

    q, scale = hadamard_transform_quant_fp8(inputs.index_query, scale=128 ** -0.5)
    weights = inputs.index_weights * (4 ** -0.5 * 128 ** -0.5) * scale.squeeze(-1)
    indexer = prefill.indexer
    for row in [258, 526 - 1]:
        length = int(meta.lengths[row])
        closed = length // 4
        indices = meta.ragged[int(meta.ks[row]) + torch.arange(closed, device="cuda") * 4 + 3].long()
        keys = indexer[indices, :128].view(torch.float8_e4m3fn).float()
        key_scale = indexer[indices, 128:].contiguous().view(torch.float32).flatten()
        logits = ((q[row].float() @ keys.T).relu() * weights[row, :, None]).sum(0) * key_scale
        groups = logits.topk(64).indices
        selected = physical[row, :256].view(64, 4)[:, 3].long()
        assert set(selected.tolist()) == set(indices[groups].tolist())
    if verify_candidates:
        raw_history = {
            2: torch.cat((inputs.index_key[:259], inputs.index_gate[:259]), -1).clone(),
            0: torch.cat((inputs.index_key[259:], inputs.index_gate[259:]), -1).clone(),
        }
        original_tail = decode.tail[3].clone()
        meta = None
        graph = None
        fixed_inputs = None
        for iteration, (rows, widths, prefixes) in enumerate(
            [([2, 0], [3, 2], [259, 267]), ([0, 2], [3, 1], [268, 260]), ([2, 0], [2, 3], [261, 269])]
        ):
            requests, steps, ready = [], [], []
            for row, width, prefix in zip(rows, widths, prefixes):
                requests += [row] * width
                steps += list(range(width))
                ready += list(range(prefix, prefix + width))
            count = len(requests)
            requests += [3] * (6 - count)
            steps += [0] * (6 - count)
            ready += [0] * (6 - count)
            incoming = data(6)
            incoming_batch = replace(
                make_batch([1] * 6, requests, ready, table, lane=1),
                hold_request_id=3,
                mtp_indices=torch.tensor(steps, device="cuda", dtype=torch.int32),
            )
            new_meta = decode.create_metadata(incoming_batch)
            locations = torch.stack([table[r, n] for r, n in zip(requests[:count], ready[:count])])
            writer(token_indices=locations, kv=torch.randn(count, 1, 512, device="cuda", dtype=dtype) * 0.2)
            offset = 0
            for row, width, prefix in zip(rows, widths, prefixes):
                raw = torch.cat(
                    (incoming.index_key[offset : offset + width], incoming.index_gate[offset : offset + width]), -1
                )
                raw_history[row] = torch.cat((raw_history[row][:prefix], raw), 0)
                offset += width
            if graph is None:
                meta, fixed_inputs = new_meta, incoming
                output = decode.run(fixed_inputs, meta)
                saved_tail = decode.tail.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = decode.run(fixed_inputs, meta)
                decode.tail.copy_(saved_tail)
            else:
                meta.copy_for_graph(new_meta)
                for name in ("query", "index_key", "index_gate", "index_query", "index_weights"):
                    getattr(fixed_inputs, name).copy_(getattr(incoming, name))
                graph.replay()
                output = captured
            physical, _ = decode.select_indices(fixed_inputs, meta)
            torch.testing.assert_close(output.float(), expected(fixed_inputs, meta, physical), rtol=0.02, atol=0.002)
            assert torch.count_nonzero(output[count:]) == 0
            for row, width, prefix in zip(rows, widths, prefixes):
                for end in range((prefix // 4 + 1) * 4, prefix + width + 1, 4):
                    raw = raw_history[row][end - 4 : end].float()
                    pooled = (raw[:, :128] * (raw[:, 128:] + params.pool_position_bias).softmax(0)).sum(0).bfloat16()
                    key, scale = hadamard_transform_quant_fp8(pooled[None], 128 ** -0.5)
                    packed = decode.indexer[table[row, end - 1].long()]
                    torch.testing.assert_close(
                        packed[:128].view(torch.float8_e4m3fn).float(), key[0].float(), rtol=0, atol=0
                    )
                    torch.testing.assert_close(packed[128:].view(torch.float32), scale.flatten(), rtol=0, atol=0)
                for position in range(prefix + width - spec.pool_size, prefix + width):
                    torch.testing.assert_close(
                        decode.tail[row, position % 6], raw_history[row][position], rtol=0, atol=0
                    )
            torch.testing.assert_close(decode.tail[3], original_tail, rtol=0, atol=0)
        return
    rows, ready = [0, 2], [267, 259]
    inputs = data(2)
    locations = torch.stack([table[r, n] for r, n in zip(rows, ready)])
    writer(token_indices=locations, kv=torch.randn(2, 1, 512, device="cuda", dtype=spec.dtype) * 0.2)
    meta = decode.create_metadata(make_batch([1, 1], rows, ready, table))
    physical, _ = decode.select_indices(inputs, meta)
    output = decode.run(inputs, meta)
    torch.testing.assert_close(output.float(), expected(inputs, meta, physical), rtol=0.02, atol=0.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode.run(inputs, meta)
    rows, ready = [2, 0], [260, 268]
    inputs.query.add_(0.01)
    new = decode.create_metadata(make_batch([1, 1], rows, ready, table))
    meta.copy_for_graph(new)
    graph.replay()
    physical, _ = decode.select_indices(inputs, meta)
    torch.testing.assert_close(captured.float(), expected(inputs, meta, physical), rtol=0.02, atol=0.002)

    # Capture below top-k, then grow across it without changing the captured path.
    short = decode.create_metadata(make_batch([1, 1], [0, 3], [6, 0], table))
    short_output = decode.run(inputs, short)
    assert torch.count_nonzero(short_output[1]) == 0
    growth_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(growth_graph):
        grown = decode.run(inputs, short)
    short.copy_for_graph(decode.create_metadata(make_batch([1, 1], rows, ready, table)))
    growth_graph.replay()
    physical, _ = decode.select_indices(inputs, short)
    torch.testing.assert_close(grown.float(), expected(inputs, short, physical), rtol=0.02, atol=0.002)


def test_kda_decode_rejects_dimensions_without_a_kernel():
    spec = KimiDeltaAttentionSpec(2, 32, 4, torch.bfloat16)
    fmt = FormatSpec(
        "conv_ssm", torch.bfloat16, "request_conv_ssm", ("conv", "ssm"), 1, field_dtypes=(torch.bfloat16, torch.float32)
    )
    caps, limits = DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(0, 2, 128)
    assert KimiDeltaAttentionBackend("prefill").probe(spec, fmt, caps, limits).offers
    assert "128-dimensional" in KimiDeltaAttentionBackend("decode").probe(spec, fmt, caps, limits).reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,quantized,mhc,moe",
    [
        ("triton", False, True, False),
        ("flashmla_sparse", False, False, False),
        ("triton", True, True, False),
        ("triton", False, True, True),
        ("flashmla_sparse", True, True, False),
        ("flashmla_sparse", True, True, True),
    ],
)
@torch.no_grad()
def test_glm_model_binding_prefix_and_graph(
    implementation, quantized, mhc, moe, monkeypatch, tmp_path, verify_layout=None, draft_mode=None
):
    from dataclasses import asdict
    import json
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel
    from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
    from lightllm.common.execution.runner.decode_graph import CudaGraph
    from lightllm.common.execution.runner.prefill_graph import PrefillCudaGraph
    from unit_tests.common.execution.test_model_cuda import prepare_model_batch

    config = dict(
        model_type="glm5_next",
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        rms_norm_eps=1e-5,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=0,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=256,
        index_kpool=4,
        linear_attn_config=dict(num_heads=2, head_dim=128, short_conv_kernel_size=4),
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mhc=mhc,
        hc_mult=4,
        n_routed_experts=4,
        n_shared_experts=1,
        first_k_dense_replace=1 if moe else 4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        swiglu_limit=10.0,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    if draft_mode:
        config["num_nextn_predict_layers"] = 1
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in (
        "GLOBAL_RANK",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
        "CURRENT_RANK_IN_DP",
        "CURRENT_RANK_IN_NODE",
        "CURRENT_DEVICE_ID",
    ):
        monkeypatch.setenv("LIGHTLLM_" + name, "0")
    for name in ("GLOBAL_WORLD_SIZE", "DP_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + name, "1")
    args = StartArgs(
        model_dir=str(tmp_path),
        data_type="bfloat16",
        page_size=1,
        llm_kv_type="fp8kv_dsa" if quantized else "None",
        llm_prefill_att_backend=[implementation, "triton"],
        llm_decode_att_backend=[implementation, "triton"],
        linear_att_ssm_data_type="float32",
        running_max_req_size=2,
        prefill_cudagraph_max_handle_token=5,
        enable_fused_shared_experts=False,
        mtp_mode=draft_mode or ("eagle_with_att" if verify_layout else None),
        mtp_step=2 if verify_layout or draft_mode else 0,
        mtp_draft_model_dir=[str(tmp_path)] * (2 if draft_mode == "vanilla_with_att" else 1)
        if verify_layout or draft_mode
        else None,
        mtp_dynamic_verify=verify_layout == "dynamic",
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    torch.manual_seed(139)
    weights = {}

    def put(name, shape, kind="normal"):
        weights[name] = (
            torch.ones(shape) if kind == "one" else torch.zeros(shape) if kind == "zero" else torch.randn(shape) * 0.05
        )

    put("model.embed_tokens.weight", (256, 128))
    put("lm_head.weight", (256, 128))
    put("model.norm.weight", (128,), "one")
    for i in range(4):
        prefix = f"model.layers.{i}"
        put(prefix + ".input_layernorm.weight", (128,), "one")
        put(prefix + ".post_attention_layernorm.weight", (128,), "one")
        if mhc:
            for family in ("attn", "ffn"):
                put(f"{prefix}.hc_{family}_fn", (24, 512))
                put(f"{prefix}.hc_{family}_base", (24,), "zero")
                put(f"{prefix}.hc_{family}_scale", (3,), "one")
        att = prefix + ".self_attn"
        if i < 3:
            for name, width in [
                ("q", 256),
                ("k", 256),
                ("v", 256),
                ("b", 2),
                ("f_a", 128),
                ("g_a", 128),
                ("f_b", 256),
                ("g_b", 256),
            ]:
                put(f"{att}.{name}_proj.weight", (width, 128))
            for name in ("q", "k", "v"):
                put(f"{att}.{name}_conv1d.weight", (256, 1, 4))
            put(att + ".o_proj.weight", (128, 256))
            put(att + ".o_norm.weight", (128,), "one")
            put(att + ".A_log", (2,))
            put(att + ".dt_bias", (256,))
        else:
            for name, shape in [
                ("q_a_proj", (128, 128)),
                ("kv_a_proj_with_mqa", (512, 128)),
                ("q_b_proj", (512, 128)),
                ("kv_b_proj", (1024, 512)),
                ("o_proj", (128, 512)),
            ]:
                put(f"{att}.{name}.weight", shape)
            put(att + ".q_a_layernorm.weight", (128,), "one")
            put(att + ".kv_a_layernorm.weight", (512,), "one")
            for name, shape in [("wq_b", (512, 128)), ("wk", (128, 128)), ("weights_proj", (4, 128))]:
                put(f"{att}.indexer.{name}.weight", shape)
            put(att + ".indexer.k_norm.weight", (128,), "one")
            put(att + ".indexer.k_norm.bias", (128,), "zero")
            put(att + ".indexer.index_kpool_compress_gate", (128, 128))
            put(att + ".indexer.index_kpool_compress_ape", (4, 128))
        if moe and i >= 1:
            put(prefix + ".mlp.gate.weight", (4, 128))
            put(prefix + ".mlp.gate.e_score_correction_bias", (4,), "zero")
            mlps = [prefix + f".mlp.experts.{j}" for j in range(4)] + [prefix + ".mlp.shared_experts"]
        else:
            mlps = [prefix + ".mlp"]
        for mlp in mlps:
            for name in ("gate_proj", "up_proj", "down_proj"):
                put(mlp + "." + name + ".weight", (128, 128))

    if draft_mode:
        weights.update(
            {
                name.replace("model.layers.3.", "model.layers.4."): value.clone()
                for name, value in list(weights.items())
                if name.startswith("model.layers.3.")
            }
        )
        for name in ("enorm", "hnorm", "shared_head.norm"):
            put(f"model.layers.4.{name}.weight", (128,), "one")
        put("model.layers.4.eh_proj.weight", (128, 256))
    drafts_by_path = {}

    def create(native):
        with monkeypatch.context() as patch:
            if not native:
                patch.setattr(Glm5NextTpPartModel, "model_adapter_class", None)
                patch.setattr(Glm5NextMTPModel, "model_adapter_class", None)
            kwargs = dict(
                run_mode="normal",
                weight_dir=str(tmp_path),
                max_total_token_num=128,
                batch_max_tokens=64,
                max_req_num=2,
                max_seq_length=64,
                graph_max_batch_size=2,
                graph_max_len_in_batch=64,
                disable_cudagraph=True,
            )
            model = Glm5NextTpPartModel(dict(kwargs, weight_dict=weights.copy()))
            drafts = []
            if draft_mode:
                for _ in range(2 if draft_mode == "vanilla_with_att" else 1):
                    drafts.append(
                        Glm5NextMTPModel(
                            dict(
                                kwargs,
                                weight_dict=weights.copy(),
                                main_model=model,
                                mtp_previous_draft_models=drafts.copy(),
                            )
                        )
                    )
            drafts_by_path[native] = drafts
        if native:
            assert len(model.moe_plan.bindings) == (3 if moe else 0) and not model.moe_plan.legacy
            for layer_id, bound in model.moe_plan.bindings:
                assert model.layers_infer[layer_id].bound_moe is bound
                monkeypatch.setattr(
                    model.trans_layers_weight[layer_id].experts,
                    "experts",
                    lambda *a, **kw: pytest.fail("native GLM called weight-loader MoE"),
                )
        for item in [model, *drafts]:
            item.prepare_attention_states()
        model.att_state_manager.finish_state_registration()
        for state in model.att_state_manager.get_full_attention_states():
            state.device = torch.device("cuda")
            state.allocate_storage(128)
        for item in [model, *drafts]:
            item.req_manager = SimpleNamespace(
                req_to_token_indexs=torch.arange(192, dtype=torch.int32, device="cuda").view(3, 64), HOLD_REQUEST_ID=2
            )
            if native:
                item.prefill_att_backend = item.decode_att_backend = None
                item.prefill_att_backend1 = item.decode_att_backend1 = None
            else:
                item._init_att_backend()
                item._init_att_backend1()
            item._init_execution_plan()
            item._init_hidden_collector()
            assert (item.execution_plan is not None) == native
        return model

    try:
        legacy, native = create(False), create(True)
        assert len(native.execution_plan.bindings) == 8
        for kind in ("get_full_attention_states", "get_bounded_attention_states"):
            old, new = getattr(legacy.att_state_manager, kind)(), getattr(native.att_state_manager, kind)()
            assert [s.config for s in old] == [s.config for s in new]
        monkeypatch.setattr(
            native.att_state_manager, "get_state", lambda *a, **k: pytest.fail("runtime manager lookup")
        )

        if draft_mode:
            from unit_tests.common.execution.test_glm_verify import exercise_glm_drafts

            exercise_glm_drafts(native, drafts_by_path[True], drafts_by_path[False], monkeypatch)
            return

        if verify_layout:
            from unit_tests.common.execution.test_model_cuda import exercise_verification

            def check_verify(output, sequences):
                actual = output.logits.clone()
                expected = []
                for sequence in sequences:
                    for state in legacy.att_state_manager.get_bounded_attention_states():
                        state.reset_bounded_att_state([0, 1])
                    batch = prepare_model_batch(legacy, [sequence, sequence], [0, 0], True)
                    expected.append(legacy._context_forward(batch).logits[:1].clone())
                target = torch.cat(expected)
                torch.testing.assert_close(actual, target, rtol=0.02, atol=0.005)
                assert torch.equal(actual.argmax(-1), target.argmax(-1))

            exercise_verification(native, lambda *a: prepare_model_batch(native, *a), check_verify, verify_layout)
            return

        def forward(model, sequences, cached, prefill):
            batch = prepare_model_batch(model, sequences, cached, prefill)
            return (model._context_forward(batch) if prefill else model._token_forward(batch)).logits.clone()

        def check(sequences, cached, prefill):
            target = forward(legacy, sequences, cached, prefill)
            actual = forward(native, sequences, cached, prefill)
            torch.testing.assert_close(actual, target, rtol=0.02, atol=0.005)
            assert torch.equal(actual.argmax(-1), target.argmax(-1))

        seq = [[11, 12, 13], [21, 22]]
        check(seq, [0, 0], True)
        seq = [seq[0] + [14], seq[1] + [23]]
        check(seq, [3, 2], False)

        def snapshot():
            views = {
                **native.attention_execution.resources.stores.get("history").views(),
                **native.attention_execution.resources.stores.get("recurrent").views(),
                **native.attention_execution.resources.stores.get("window").views(),
            }
            return [(t, t.clone()) for view in views.values() for t in view.fields]

        saved = snapshot()
        native.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=64)
        batch = prepare_model_batch(native, seq, [3, 2], False)
        batch.is_cuda_graph_capture = True
        captured = native.graph.capture_decode(native._token_forward, batch)
        for tensor, value in saved:
            tensor.copy_(value)
        seq = [seq[0] + [15], seq[1] + [24]]
        target = forward(legacy, seq, [4, 3], False)
        batch = prepare_model_batch(native, seq, [4, 3], False)
        batch.is_cuda_graph_replay = True
        native.graph.replay(batch)
        torch.testing.assert_close(captured.logits, target, rtol=0.02, atol=0.005)
        seq = [seq[0] + [16, 17], seq[1] + [25, 26]]
        check(seq, [5, 4], True)
        native.prefill_graph = PrefillCudaGraph(native.graph, 1)
        seq = [seq[0] + [18, 19], seq[1] + [27, 28]]
        check(seq, [7, 6], True)
        seq = [seq[0] + [20, 21], seq[1] + [29, 30]]
        check(seq, [9, 8], True)
    finally:
        get_env_start_args.cache_clear()
        get_llm_data_type.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pooled_index_quant_matches_torch_hadamard(dtype):
    from lightllm.common.execution.formats.kv.kernels.index_quant import hadamard_transform_quant_fp8

    torch.manual_seed(151)
    x = torch.randn(9, 4, 128, device="cuda", dtype=dtype)
    matrix = torch.ones(1, 1, device="cuda")
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    transformed = (x.float() @ matrix * 128 ** -0.5).bfloat16().float()
    scale = torch.exp2(torch.ceil(torch.log2(transformed.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    expected = (transformed / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    actual, actual_scale = hadamard_transform_quant_fp8(x, 128 ** -0.5)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scale, scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantized,implementation", [(False, "triton"), (True, "triton"), (True, "flashmla_sparse")])
@torch.no_grad()
def test_pooled_long_decode_topk_and_lane_graph(quantized, implementation):
    from lightllm.common.execution.formats.kv.kernels.index_quant import hadamard_transform_quant_fp8

    torch.manual_seed(157)
    spec, state, tail, fmt, window = pooled_contracts(quantized)
    spec = replace(spec, q_heads=64, index_heads=64, index_topk=2048)
    manager = AttentionStateManager(4)
    store = PackedMLAStateStore((state,), fmt, manager)
    windows = IndexerWindowStateStore((tail,), window, manager, 4, "cuda")
    manager.finish_state_registration()
    resource = store.resources[state.key][1]
    resource.device = torch.device("cuda")
    resource.allocate_storage(16384)
    views = {**store.views(), **windows.views()}
    params = PooledNoPEStaticParams(
        spec, lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"), torch.zeros(4, 128, device="cuda")
    )
    planner = Planner(
        (PooledNoPEAttentionBackend(implementation, "decode"),),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(64 << 20, 4, 4096, 2),
    )
    prepared = planner.prepare(
        [AttentionRequest(3, "decode", spec, state, fmt, params, auxiliary_states=(("indexer_tail", tail, window),))]
    )
    bound = prepared.bind(views, prepared.reserve(lambda n: torch.empty(n, device="cuda", dtype=torch.uint8)))
    decode = bound.bindings[0].execution
    decode.tail.zero_()
    table = torch.randperm(16384, device="cuda", dtype=torch.int32).view(4, 4096)
    kv = torch.randn(16384, 1, 512, device="cuda", dtype=spec.dtype) * 0.2
    PooledNoPEKVWriter(views[state.key])(token_indices=torch.arange(16384, device="cuda", dtype=torch.int32), kv=kv)
    decode.indexer[:, :128].copy_(
        torch.randn(decode.indexer.shape[0], 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    )
    decode.indexer[:, 128:].view(torch.float32).fill_(0.01)
    inputs = PooledNoPEInputs(
        torch.randn(2, 64, 512, device="cuda", dtype=spec.dtype) * 0.2,
        torch.randn(2, 128, device="cuda", dtype=spec.dtype) * 0.3,
        torch.randn(2, 128, device="cuda", dtype=spec.dtype) * 0.3,
        torch.randn(2, 64, 128, device="cuda", dtype=spec.dtype) * 0.3,
        torch.randn(2, 64, device="cuda"),
    )

    def check(output, rows, lengths):
        q, scale = hadamard_transform_quant_fp8(inputs.index_query, scale=128 ** -0.5)
        weights = inputs.index_weights * (64 ** -0.5 * 128 ** -0.5) * scale.squeeze(-1)
        physical, _ = decode.select_indices(inputs, meta)
        for i, (row, length) in enumerate(zip(rows, lengths)):
            closed = length // 4
            ends = table[row, 3 : closed * 4 : 4].long()
            keys = decode.indexer[ends, :128].view(torch.float8_e4m3fn).float()
            scales = decode.indexer[ends, 128:].contiguous().view(torch.float32).flatten()
            logits = ((q[i].float() @ keys.T).relu() * weights[i, :, None]).sum(0) * scales
            groups = logits.topk(512).indices
            positions = (groups[:, None] * 4 + torch.arange(4, device="cuda")).flatten()
            positions = torch.cat((positions, torch.arange(closed * 4, length, device="cuda")))
            indices = table[row, positions].long()
            assert set(physical[i][physical[i] >= 0].tolist()) == set(indices.tolist())
            cached = decode.cache[indices, 0]
            if quantized:
                payload = cached[:, :448].view(torch.float8_e4m3fn).float()
                scales = torch.exp2(cached[:, 576:583].float() - 127).repeat_interleave(64, -1)
                cached = torch.cat((payload * scales, cached[:, 448:576].view(torch.bfloat16).float()), -1)
            scores = inputs.query[i].float() @ cached.float().T * spec.score_scale
            expected = scores.softmax(-1) @ cached.float()
            torch.testing.assert_close(output[i].float(), expected, rtol=0.02, atol=0.002)

    rows, lengths = [2, 0], [3955, 3081]
    meta = decode.create_metadata(make_batch([1, 1], rows, [n - 1 for n in lengths], table))
    check(decode.run(inputs, meta), rows, lengths)
    original = meta.ragged.clone()
    lane = decode.create_metadata(make_batch([1, 1], [1, 3], [23, 0], table, lane=1))
    assert meta.ragged.data_ptr() != lane.ragged.data_ptr()
    assert torch.equal(meta.ragged, original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = decode.run(inputs, meta)
    rows, lengths = [0, 2], [4093, 3987]
    inputs.query.add_(0.01)
    meta.copy_for_graph(decode.create_metadata(make_batch([1, 1], rows, [n - 1 for n in lengths], table)))
    graph.replay()
    check(output, rows, lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads,count", [(8, 1), (8, 5), (64, 1), (64, 3), (64, 12), (16, 4)])
@torch.no_grad()
def test_pooled_fp8_prefill_workspace_and_hold(heads, count, monkeypatch):
    from lightllm.common.execution.compute.attention.pooled_nope import prefill_fp8 as kernels

    torch.manual_seed(163)
    spec, state, tail, fmt, window = pooled_contracts(True)
    spec = replace(spec, q_heads=heads, index_topk=2048)
    manager = AttentionStateManager(2)
    store = PackedMLAStateStore((state,), fmt, manager)
    windows = IndexerWindowStateStore((tail,), window, manager, 2, "cuda")
    manager.finish_state_registration()
    resource = store.resources[state.key][1]
    resource.device = torch.device("cuda")
    resource.allocate_storage(65536)
    views = {**store.views(), **windows.views()}
    params = PooledNoPEStaticParams(
        spec, lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"), torch.zeros(4, 128, device="cuda")
    )
    backend = PooledNoPEAttentionBackend("flashmla_sparse", "prefill")
    limits = ExecutionLimits(64 << 20, 2, 65536, 2)
    request = AttentionRequest(
        3, "prefill", spec, state, fmt, params, auxiliary_states=(("indexer_tail", tail, window),)
    )
    planner = Planner((backend,), DeviceCaps("cuda", 0, (9, 0)), limits)
    prepared = planner.prepare([request])
    with pytest.raises(ValueError, match="budget"):
        Planner((backend,), planner.device, replace(limits, workspace_bytes=prepared.workspace_bytes - 1)).prepare(
            [request]
        )
    workspaces = prepared.reserve(lambda n: torch.empty(n, device="cuda", dtype=torch.uint8))
    bound = prepared.bind(views, workspaces).bindings[0].execution
    monkeypatch.setattr(manager, "get_state", lambda *a, **k: pytest.fail("runtime manager lookup"))
    table = torch.full((2, 65536), 65536, device="cuda", dtype=torch.int32)
    table[0] = torch.randperm(65536, device="cuda", dtype=torch.int32)
    history = torch.randn(65537, 1, 512, device="cuda", dtype=spec.dtype) * 0.2
    PooledNoPEKVWriter(views[state.key])(
        token_indices=torch.arange(65537, device="cuda", dtype=torch.int32), kv=history
    )
    real_count = count if count == 1 else count - 1
    lengths, rows, ready = [real_count], [0], [65536 - real_count]
    if count > 1:
        lengths, rows, ready = lengths + [1], rows + [1], ready + [0]
    batch = make_batch(lengths, rows, ready, table)
    meta = bound.create_metadata(batch)
    original = meta.ragged.clone()
    lane = bound.create_metadata(replace(batch, microbatch_index=1))
    assert lane.ragged.data_ptr() != meta.ragged.data_ptr()
    assert torch.equal(original, meta.ragged)
    if count > 1:
        assert meta.ragged[65536].item() == 65536
    for name, tensor in bound.fp8_scratch[0].items():
        other = bound.fp8_scratch[1][name]
        assert tensor.data_ptr() + tensor.numel() * tensor.element_size() <= other.data_ptr()

    relative = torch.full((count, 2176), -1, device="cuda", dtype=torch.int32)
    physical = torch.full_like(relative, -1)
    fresh = torch.randn(count, 1, 512, device="cuda", dtype=spec.dtype) * 0.4
    query = torch.randn(count, heads, 512, device="cuda", dtype=spec.dtype) * 0.3
    expected = torch.zeros_like(query, dtype=torch.float32)
    for i in range(real_count):
        old = torch.arange(2048 - i - 1, device="cuda")
        new = torch.arange(i + 1, device="cuda") + ready[0]
        selected = torch.cat((old, new))
        relative[i, :2048] = selected.int()
        physical[i, :2048] = table[0, selected]
        packed = bound.cache[table[0, old].long(), 0]
        payload = packed[:, :448].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(packed[:, 576:583].float() - 127).repeat_interleave(64, -1)
        decoded = torch.cat((payload * scales, packed[:, 448:576].view(torch.bfloat16).float()), -1)
        kv = torch.cat((decoded, fresh[: i + 1, 0].float()))
        expected[i] = (query[i].float() @ kv.T * spec.score_scale).softmax(-1) @ kv
    inputs = PooledNoPEInputs(query, None, None, None, None, fresh)
    monkeypatch.setattr(bound, "select_indices", lambda *args: (physical, relative))
    calls = []
    if heads == 8 or (heads == 64 and count == 1):
        name, argument = "_context_attention_fwd_fp8_direct_smallq", "direct_workspace"
    elif heads == 64 and count <= 8:
        name, argument = "_context_attention_fwd_fp8_selected_prefill", "selected_workspace"
    else:
        name = None
    if name is not None:
        original_kernel = getattr(kernels, name)

        def execute(*args, **kwargs):
            scratch = kwargs.get(argument, args[-1])
            assert scratch is not None
            calls.append(name)
            return original_kernel(*args, **kwargs)

        monkeypatch.setattr(kernels, name, execute)
    actual = bound.run(inputs, meta)
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.002)
    if name is not None:
        assert calls == [name]
