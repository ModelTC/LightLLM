"""Native sparse MLA staging must match independent decoded-KV attention."""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4_mla import quantize_nvfp4_mla
from lightllm.models.glm5_next.triton_kernel.nvfp4_prefill import nvfp4_cached_prefill, nvfp4_sparse_prefill
from unit_tests.common.att_resources.test_nvfp4_mla import dequantize, reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("lengths", [(257,), (0, 257, 3, 512, 0, 129), (0, 1, 257) * 49])
def test_prefill_slots_maps_request_boundaries_and_replays_changed_pages(lengths):
    from lightllm.models.glm5_next.triton_kernel.nvfp4_prefill import prefill_slots

    torch.manual_seed(53)
    requests = torch.randperm(len(lengths), device="cuda", dtype=torch.int32)
    table = torch.randint(0, 4096, (len(lengths), max(lengths)), device="cuda", dtype=torch.int32)
    bounds = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    total = sum(lengths)
    prefill_slots(table, requests, bounds, total)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = prefill_slots(table, requests, bounds, total)
    table.add_(17)
    graph.replay()
    expected = torch.cat([table[requests[row], :length] for row, length in enumerate(lengths)])
    assert torch.equal(actual, expected)


def test_paired_decode_all_fp4_codes_and_positive_fp8_scales():
    from lightllm.models.glm5_next.triton_kernel.nvfp4_selected import _gather_selected

    # All 256 packed bytes cover every pair of FP4 codes. Positive finite
    # E4M3 encodings are 0..126, including zero and subnormal scales.
    packed = torch.empty((127, 1, 288), dtype=torch.uint8, device="cuda")
    packed[..., :256] = torch.arange(256, device="cuda").to(torch.uint8)
    packed[..., 256:] = torch.arange(127, device="cuda").to(torch.uint8).view(-1, 1, 1)
    indices = torch.arange(128, dtype=torch.int32, device="cuda").view(1, -1)
    lengths = torch.tensor([127], dtype=torch.int32, device="cuda")
    output = torch.empty((128, 1, 512), dtype=torch.bfloat16, device="cuda")
    _gather_selected[(1, 32)](packed, indices, lengths, None, None, output, 288, (0, 0, 0), 128, False, 4)
    assert torch.equal(output[:127].view(torch.int16), dequantize(packed).view(torch.int16))


@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("workspace_tokens", [64, 1024])
@pytest.mark.parametrize("graph", [False, True])
def test_sparse_staging_matches_decoded_reference_with_current_override_and_empty_chunks(
    heads, workspace_tokens, graph
):
    torch.manual_seed(748)
    rows, tokens = 17, 769
    raw = torch.randn(1024, 1, 512, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty(1024, 1, 288, dtype=torch.uint8, device="cuda")
    quantize_nvfp4_mla(raw, torch.arange(1024, device="cuda", dtype=torch.int32), packed)
    ragged = torch.randperm(1024, device="cuda", dtype=torch.int32)[:tokens]
    current_slots = ragged[-7:].long()
    current = raw[current_slots].clone().add_(0.125)
    row_table = torch.full((1024,), -1, device="cuda", dtype=torch.int32)
    row_table[current_slots] = torch.arange(7, device="cuda", dtype=torch.int32)
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    relative = torch.randint(0, tokens, (rows, 2176), device="cuda", dtype=torch.int32)
    relative[0] = -1
    relative[1] %= 3
    relative[2] = tokens - 1
    relative[3, ::3] = -1
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)

    def run():
        if workspace_tokens == 1024:
            physical = ragged[relative.clamp_min(0).long()].masked_fill(relative < 0, -1)
            return nvfp4_cached_prefill(q, packed, ragged, starts, relative, physical, 0.0625, current, row_table)
        return nvfp4_sparse_prefill(
            q, packed, ragged, starts, relative, 0.0625, current, row_table, workspace_bytes=workspace_tokens * 1024
        )

    out = run()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            out = run()
        relative[4, :] = tokens - 2
        captured.replay()
    physical = ragged[relative.clamp_min(0).long()].masked_fill(relative < 0, -1)
    decoded = dequantize(packed)
    decoded[current_slots] = current
    expected = reference(q, decoded, physical, 0.0625)
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.016)
    assert torch.isfinite(out).all() and not out[0].any()


@pytest.mark.parametrize("rows", [17, 65, 513])
@pytest.mark.parametrize("current", [False, True])
def test_selected_staging_is_context_independent_and_graph_safe(rows, current):
    from lightllm.models.glm5_next.triton_kernel.nvfp4_selected import nvfp4_selected_attention

    torch.manual_seed(741)
    raw = torch.randn(4096, 1, 512, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty(4096, 1, 288, dtype=torch.uint8, device="cuda")
    quantize_nvfp4_mla(raw, torch.arange(4096, device="cuda", dtype=torch.int32), packed)
    q = torch.randn(rows, 16, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 4096, (rows, 2176), device="cuda", dtype=torch.int32)
    indices[0] = -1
    indices[1, ::3] = -1
    row_table = torch.full((4096,), -1, device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 100, 3998, 1729, 72, 91, 2941], device="cuda", dtype=torch.int32)
    row_table[slots.long()] = torch.arange(7, device="cuda", dtype=torch.int32)
    raw_current = raw[slots.long()].clone().add_(0.125) if current else None

    def run():
        return nvfp4_selected_attention(q, packed, indices, 0.0625, raw_current, row_table if current else None)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    indices[-1] = 3
    graph.replay()
    decoded = dequantize(packed)
    if current:
        decoded[slots.long()] = raw_current
    expected = reference(q, decoded, indices, 0.0625)
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.016)
    assert torch.isfinite(out).all() and not out[0].any()


def test_multirequest_staging_graph_spans_disjoint_workspace_chunks():
    torch.manual_seed(744)
    raw = torch.randn(1024, 1, 512, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty(1024, 1, 288, dtype=torch.uint8, device="cuda")
    quantize_nvfp4_mla(raw, torch.arange(1024, device="cuda", dtype=torch.int32), packed)
    ragged = torch.randperm(1024, device="cuda", dtype=torch.int32)[:513]
    q = torch.randn(17, 16, 512, device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0] * 8 + [256] * 9, device="cuda", dtype=torch.int32)
    relative = torch.randint(0, 256, (17, 2176), device="cuda", dtype=torch.int32)
    relative[0] = -1
    slots = ragged[torch.tensor([1, 255, 257, 400], device="cuda")].long()
    current = raw[slots].clone().add_(0.125)
    row_table = torch.full((1024,), -1, device="cuda", dtype=torch.int32)
    row_table[slots] = torch.arange(4, device="cuda", dtype=torch.int32)

    def run():
        return nvfp4_sparse_prefill(
            q, packed, ragged, starts, relative, 0.0625, current, row_table, workspace_bytes=128 * 1024
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    relative[-1] = 1
    graph.replay()
    physical = ragged[(relative.clamp_min(0) + starts[:, None]).long()].masked_fill(relative < 0, -1)
    decoded = dequantize(packed)
    decoded[slots] = current
    torch.testing.assert_close(out, reference(q, decoded, physical, 0.0625), rtol=0.02, atol=0.016)


@pytest.mark.parametrize("rows", [1, 8, 16, 65, 129, 513])
def test_decode_reuses_causal_lengths_and_readonly_metadata(rows):
    from lightllm.models.glm5_next.triton_kernel.nvfp4_selected import (
        nvfp4_selected_attention,
        prepare_selected_metadata,
    )

    torch.manual_seed(7481)
    raw = torch.randn(4096, 1, 512, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty(4096, 1, 288, dtype=torch.uint8, device="cuda")
    quantize_nvfp4_mla(raw, torch.arange(4096, device="cuda", dtype=torch.int32), packed)
    q = torch.randn(rows, 16, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, 4096, (rows, 2176), device="cuda", dtype=torch.int32)
    lengths = (torch.arange(rows, device="cuda", dtype=torch.int32) * 127) % 2052
    indices.masked_fill_(torch.arange(2176, device="cuda")[None, :] >= lengths[:, None], -1)
    table = torch.arange(512 * 2176, device="cuda", dtype=torch.int32).view(512, 2176)
    cu_q = torch.arange(513, device="cuda", dtype=torch.int32)

    def run():
        metadata, offsets = prepare_selected_metadata(lengths, q.shape[1], indices.shape[1], cu_q)
        kwargs = dict(
            cache_lengths=lengths,
            page_table=table,
            cu_q=cu_q,
            scheduler_metadata=metadata,
            scheduler_offsets=offsets,
        )
        first = nvfp4_selected_attention(q, packed, indices, 0.0625, **kwargs)
        second = nvfp4_selected_attention(q, packed, indices, 0.0625, **kwargs)
        return first, second

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, reused = run()
    lengths.fill_(2049)
    indices.random_(0, 4096)
    indices[:, 2049:] = -1
    graph.replay()
    expected = nvfp4_selected_attention(q, packed, indices, 0.0625)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(reused, expected, rtol=0, atol=0)
    assert torch.equal(table.flatten(), torch.arange(table.numel(), device="cuda", dtype=torch.int32))
    assert torch.equal(cu_q, torch.arange(513, device="cuda", dtype=torch.int32))
