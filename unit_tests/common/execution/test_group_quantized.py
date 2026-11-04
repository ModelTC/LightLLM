from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention_state import AttentionStateManager
from lightllm.core.compute.attention.dense.backend import DenseStaticParams
from lightllm.core.compute.attention.dense.group_quantized import GroupQuantizedAttentionBackend
from lightllm.core.formats.kv.group_quantized import GroupQuantizedKVWriter
from lightllm.core.planner.planner import Planner
from lightllm.common.execution.runner import attention_batch
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec, StateSpec
from lightllm.core.storage.attention import DenseStateStore


def make_format(bits, dtype=torch.float16, group=8, page=4):
    return FormatSpec(
        f"int{bits}",
        torch.int8,
        "token_head_dim",
        ("k", "v", "k_scale", "v_scale"),
        page,
        group_size=group,
        scale_dtype=dtype,
    )


def make_bound(bits, dtype, group, phase, window=(-1, -1), device="cuda"):
    manager = AttentionStateManager(2)
    state_spec = StateSpec("main", "kv", 0, (0,), (2, 64))
    spec = DenseAttentionSpec(8, 2, 64, window=window, dtype=dtype)
    store = DenseStateStore((state_spec,), make_format(bits, dtype, group), manager)
    manager.finish_state_registration()
    state = manager.get_full_attention_states()[0]
    state.device = torch.device(device)
    state.allocate_storage(2048)
    view = store.views()[state_spec.key]
    planner = Planner(
        (GroupQuantizedAttentionBackend(phase),),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(8 * 1024 ** 2, 2, 768, 2),
    )
    alloc = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
    plan = planner.bind(
        [(0, phase, spec, view, DenseStaticParams(spec, alloc), "triton")], lambda n: alloc(n, torch.uint8)
    )
    return plan.bindings[0].execution, GroupQuantizedKVWriter(view), state


@pytest.mark.parametrize("bits", [4, 8])
def test_group_format_is_bound_without_model_specific_registration(bits):
    bound, writer, state = make_bound(bits, torch.float16, 32, "decode", device="cpu")
    assert bound.view.format.group_size == state.config.group_size == 32
    assert bound.k.data_ptr() == state.k_storage[0].data_ptr()
    assert bound.k_scale.data_ptr() == state.k_scales[0].data_ptr()
    assert writer.k_scale.data_ptr() == bound.k_scale.data_ptr()
    assert bound._lane_workspace(0).data_ptr() != bound._lane_workspace(1).data_ptr()


def test_probe_checks_quantization_and_query_dtype_before_allocation():
    backend = GroupQuantizedAttentionBackend("prefill")
    caps, limits = DeviceCaps("cuda"), ExecutionLimits(0)
    spec, format = DenseAttentionSpec(4, 2, 64, dtype=torch.bfloat16), make_format(8)
    assert backend.probe(spec, format, caps, limits).offers
    for invalid in (
        replace(format, fields=("k", "v")),
        replace(format, dtype=torch.float16),
        replace(format, scale_dtype=torch.int8),
        replace(format, group_size=128),
    ):
        assert backend.probe(spec, invalid, caps, limits).reason
    assert backend.probe(replace(spec, dtype=torch.float32), format, caps, limits).reason


def unpack(data, scale, bits, group, dtype):
    values = data.float()
    if bits == 4:
        values = torch.stack(((data.int() & 15) - 7, ((data.int() >> 4) & 15) - 7), dim=-1)
        values = values.flatten(-2).float()
    return (values * scale.float().repeat_interleave(group, dim=-1)).to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group", [8, 32])
@pytest.mark.parametrize(
    "phase,window", [("prefill", (-1, -1)), ("decode", (-1, -1)), ("prefill", (63, 0)), ("decode", (63, 0))]
)
def test_quantized_attention_matches_decoded_sdpa_with_graph_and_lanes(bits, dtype, group, phase, window):
    torch.manual_seed(77)
    bound, writer, state = make_bound(bits, dtype, group, phase, window)
    # Use noncontiguous logical token positions and independently decode packed storage.
    ids = torch.randperm(2048, device="cuda")[:1536].int()
    kv = torch.randn(1536, 4, 64, dtype=dtype, device="cuda")
    writer(token_indices=ids, kv=kv)
    table = ids.view(2, 768)
    key = unpack(bound.k, bound.k_scale, bits, group, dtype)
    value = unpack(bound.v, bound.v_scale, bits, group, dtype)
    del state
    q_len = 3 if phase == "prefill" else 1

    def metadata(lengths, lane):
        return bound.create_metadata(
            attention_batch(
                SimpleNamespace(
                    batch_size=2,
                    max_q_seq_len=q_len,
                    max_kv_seq_len=max(lengths),
                    microbatch_index=lane,
                    req_manager=SimpleNamespace(req_to_token_indexs=table),
                    b_req_idx=torch.arange(2, device="cuda", dtype=torch.int32),
                    b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
                    b_q_start_loc=torch.tensor([0, q_len], device="cuda", dtype=torch.int32),
                    b_ready_cache_len=torch.tensor(lengths, device="cuda", dtype=torch.int32) - q_len,
                    b1_cu_q_seq_len=torch.tensor([0, q_len, q_len * 2], device="cuda", dtype=torch.int32),
                    b1_cu_kv_seq_len=torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32),
                )
            )
        )

    def reference(q, lengths):
        outputs = []
        for i, length in enumerate(lengths):
            q_pos = torch.arange(length - q_len, length, device="cuda")[:, None]
            k_pos = torch.arange(length, device="cuda")[None, :]
            mask = q_pos >= k_pos
            if window != (-1, -1):
                mask &= q_pos - k_pos <= window[0]
            output = F.scaled_dot_product_attention(
                q[i * q_len : (i + 1) * q_len].transpose(0, 1),
                key[table[i, :length]].transpose(0, 1),
                value[table[i, :length]].transpose(0, 1),
                attn_mask=mask,
                enable_gqa=True,
            )
            outputs.append(output.transpose(0, 1))
        return torch.cat(outputs)

    lengths = [513, 65]
    metas = [metadata(lengths, lane) for lane in range(2)]
    queries = [torch.randn(2 * q_len, 8, 64, dtype=dtype, device="cuda") for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    results = []
    for stream, q, meta in zip(streams, queries, metas):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            results.append(bound.run(q, meta))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for q, result in zip(queries, results):
        torch.testing.assert_close(result, reference(q, lengths), atol=5e-3, rtol=2e-2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound.run(queries[0], metas[0])
    lengths = [517, 68]
    metas[0].copy_for_graph(metadata(lengths, 0))
    queries[0].add_(0.1)
    graph.replay()
    torch.testing.assert_close(captured, reference(queries[0], lengths), atol=5e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group", [8, 32])
def test_writer_bytes_and_scales_match_independent_quantization(bits, dtype, group):
    torch.manual_seed(11)
    bound, writer, _ = make_bound(bits, dtype, group, "prefill")
    ids = torch.tensor([7, 3, 1025, 511, 1024], device="cuda", dtype=torch.int64)
    inputs = torch.randn(5, 4, 64, dtype=dtype, device="cuda")
    inputs[0, :, :group] = 0
    inputs[1, :, :group] = torch.finfo(dtype).tiny / 32
    inputs[2, :, :group] = torch.finfo(dtype).tiny * 0.5
    inputs[3, :, :group] = torch.finfo(dtype).tiny * 16
    writer(token_indices=ids, kv=inputs)
    for values, actual, scales in ((inputs[:, :2], bound.k, bound.k_scale), (inputs[:, 2:], bound.v, bound.v_scale)):
        groups = values.float().reshape(5, 2, 64 // group, group)
        limit = 7 if bits == 4 else 127
        expected_scale = (groups.abs().amax(-1) / limit).to(dtype)
        if dtype == torch.bfloat16:
            # The stored format flushes BF16 subnormal scales to zero.
            expected_scale = torch.where(expected_scale < torch.finfo(dtype).tiny, 0, expected_scale)
        divisor = expected_scale.float().unsqueeze(-1)
        encoded = torch.where(divisor == 0, 0, groups / divisor).trunc().clamp(-limit, limit)
        encoded = encoded.to(torch.int8).reshape(5, 2, 64)
        if bits == 4:
            encoded = encoded.clamp(-7, 7) + 7
            encoded = (encoded[..., ::2] | (encoded[..., 1::2] << 4)).to(torch.int8)
        torch.testing.assert_close(actual[ids], encoded, rtol=0, atol=0)
        torch.testing.assert_close(scales[ids], expected_scale, rtol=0, atol=0)
