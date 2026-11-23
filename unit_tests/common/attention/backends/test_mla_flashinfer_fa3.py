"""FlashInfer and FA3 MLA backends: numerics, CUDA graph step-builder state and supported layer configs."""

import pytest
import torch

from lightllm.common.attention.backend import AttentionMetadata
from lightllm.common.attention.backends.mla import (
    MLADecodeLayerConfig,
    MLADecodeLayerInputs,
    MLAPrefillLayerConfig,
    MLAPrefillLayerInputs,
)
from lightllm.common.attention.backends.mla.fa3 import FA3MLADecodeBackend, FA3MLAPrefillBackend
from lightllm.common.attention.backends.mla.flashinfer import (
    FlashInferMLABuilderConfig,
    FlashInferMLADecodeBackend,
    FlashInferMLAPrefillBackend,
)
from lightllm.common.attention.backends.mla.triton import TritonMLADecodeBackend, TritonMLAPrefillBackend

RANK, ROPE, NOPE, V_DIM, SCALE = 512, 64, 128, 128, 0.1


def prefill_config(heads=16, causal=True):
    return MLAPrefillLayerConfig(heads, NOPE, ROPE, V_DIM, SCALE, torch.bfloat16, causal=causal)


def decode_config(heads=16, causal=True):
    return MLADecodeLayerConfig(heads, RANK, ROPE, SCALE, torch.bfloat16, causal=causal)


def test_non_causal_mla_is_rejected_except_by_fa3():
    for backend in (TritonMLAPrefillBackend, FlashInferMLAPrefillBackend):
        with pytest.raises(NotImplementedError, match="causal"):
            backend.get_impl_cls()(prefill_config(causal=False))
    for backend in (TritonMLADecodeBackend, FlashInferMLADecodeBackend):
        with pytest.raises(NotImplementedError, match="causal"):
            backend.get_impl_cls()(decode_config(causal=False))
    FA3MLAPrefillBackend.get_impl_cls()(prefill_config(causal=False))
    FA3MLADecodeBackend.get_impl_cls()(decode_config(causal=False))


# ---- GPU ----

pytestmark_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def paged_token_map(requests, page_size, max_len, seed=0):
    pages = torch.randperm(requests * max_len // page_size, device="cuda", dtype=torch.int32, generator=gen(seed))
    pages = pages.view(requests, -1)
    return (pages[:, :, None] * page_size + torch.arange(page_size, device="cuda")).reshape(requests, max_len)


def make_batch(token_map, req_ids, lengths, *, prefill=False, ready=None, rows=1, capacity=None, microbatch=0):
    q_lens = [n - r for n, r in zip(lengths, ready)] if prefill else [1] * len(lengths)
    cu = lambda xs: torch.tensor([0, *xs], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)  # noqa: E731
    return AttentionMetadata(
        req_to_token_indexs=token_map,
        b_req_idx=torch.tensor(req_ids, device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        cu_q_seq_len=cu(q_lens),
        cu_kv_seq_len=cu(lengths),
        b_q_start_loc=cu(q_lens)[:-1],
        b_ready_cache_len=torch.tensor(ready or [n - 1 for n in lengths], device="cuda", dtype=torch.int32),
        max_q_seq_len=max(q_lens),
        decode_rows_per_request=rows,
        pad_req_idx=7,
        cuda_graph_max_kv_seq_len=capacity,
        microbatch_index=microbatch,
    )


def prefill_reference(q, k_nope, k_rope, v, lengths, ready, causal=True):
    out, q_off, kv_off = [], 0, 0
    for length, cached in zip(lengths, ready):
        count = length - cached
        keys = torch.cat(
            (k_nope[kv_off : kv_off + length], k_rope[kv_off : kv_off + length].expand(-1, q.shape[1], -1)), -1
        )
        scores = torch.einsum("qhd,khd->hqk", q[q_off : q_off + count].float(), keys.float()) * SCALE
        if causal:
            mask = torch.arange(length, device="cuda")[None, :] <= torch.arange(cached, length, device="cuda")[:, None]
            scores = scores.masked_fill(~mask[None], float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[kv_off : kv_off + length].float()))
        q_off, kv_off = q_off + count, kv_off + length
    return torch.cat(out)


def decode_reference(q_nope, q_rope, cache, token_map, req_ids, lengths):
    out = []
    for row, (req_id, length) in enumerate(zip(req_ids, lengths)):
        kv = cache[token_map[req_id, :length].long(), 0].float()
        scores = (q_nope[row].float() @ kv[:, :RANK].T + q_rope[row].float() @ kv[:, RANK:].T) * SCALE
        out.append(scores.softmax(-1) @ kv[:, :RANK])
    return torch.stack(out)


def prefill_inputs(lengths, ready, heads=16):
    rows = sum(n - r for n, r in zip(lengths, ready))
    kv_tokens = sum(lengths)
    return MLAPrefillLayerInputs(
        q=torch.randn(rows, heads, NOPE + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(2)),
        k_nope=torch.randn(kv_tokens, heads, NOPE, device="cuda", dtype=torch.bfloat16, generator=gen(3)),
        k_rope=torch.randn(kv_tokens, 1, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(4)),
        v=torch.randn(kv_tokens, heads, V_DIM, device="cuda", dtype=torch.bfloat16, generator=gen(5)),
    )


def builder_config(heads, page_size):
    return FlashInferMLABuilderConfig(heads, NOPE, ROPE, V_DIM, RANK, page_size, torch.bfloat16, SCALE)


# FlashInfer decode step builders keep page-index buffers for batches of up to 8 requests.
GRAPH_BUFFERS = dict(max_batch_size=8, device="cuda")


@pytestmark_cuda
@pytest.mark.parametrize("backend", ["flashinfer", "fa3"])
def test_prefill_with_cached_prefix_matches_reference(backend):
    lengths, ready = [33, 9, 259], [16, 5, 128]
    batch = make_batch(paged_token_map(3, 1, 512), [2, 0, 1], lengths, prefill=True, ready=ready)
    if backend == "flashinfer":
        pytest.importorskip("flashinfer")
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        builder = FlashInferMLAPrefillBackend.get_step_builder_cls()(builder_config(16, 1), workspace=lambda: workspace)
        impl = FlashInferMLAPrefillBackend.get_impl_cls()(prefill_config())
    else:
        builder = FA3MLAPrefillBackend.get_step_builder_cls()()
        impl = FA3MLAPrefillBackend.get_impl_cls()(prefill_config())
    inputs = prefill_inputs(lengths, ready)
    out = torch.empty(inputs.q.shape[0], 16, V_DIM, device="cuda", dtype=torch.bfloat16)
    assert impl.forward(builder.build(batch), inputs, out=out) is out
    expected = prefill_reference(inputs.q, inputs.k_nope, inputs.k_rope, inputs.v, lengths, ready)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)


@pytestmark_cuda
@pytest.mark.parametrize("backend,page_size", [("flashinfer", 1), ("flashinfer", 16), ("fa3", 1), ("fa3", 16)])
def test_decode_matches_reference(backend, page_size):
    token_map = paged_token_map(8, page_size, 512)
    cache = torch.randn(8 * 512, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(1))
    req_ids, lengths = [2, 0, 1, 3], [34, 10, 260, 512]
    batch = make_batch(token_map, req_ids, lengths)
    if backend == "flashinfer":
        pytest.importorskip("flashinfer")
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        builder = FlashInferMLADecodeBackend.get_step_builder_cls()(
            builder_config(16, page_size), workspace=lambda: workspace, max_seq_len=512, **GRAPH_BUFFERS
        )
        impl = FlashInferMLADecodeBackend.get_impl_cls()(decode_config())
    else:
        storage = torch.empty(4 * 512, dtype=torch.int32, device="cuda")
        builder = FA3MLADecodeBackend.get_step_builder_cls()(
            page_size=page_size,
            page_table_view=lambda rows, width, mb: storage[: rows * -(-width // page_size)].view(rows, -1),
        )
        impl = FA3MLADecodeBackend.get_impl_cls()(decode_config())
    q_nope = torch.randn(4, 16, RANK, device="cuda", dtype=torch.bfloat16, generator=gen(2))
    q_rope = torch.randn(4, 16, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(3))
    out = torch.empty_like(q_nope)
    impl.forward(builder.build(batch), MLADecodeLayerInputs(q_nope, q_rope, cache), out=out)
    expected = decode_reference(q_nope, q_rope, cache, token_map, req_ids, lengths)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)


@pytestmark_cuda
@pytest.mark.parametrize("page_size", [1, 3, 4, 16, 64, 128])
@pytest.mark.parametrize("lengths,num_heads", [([5, 64, 65], 32), ([1000, 32768], 32), ([1000, 131073], 128)])
def test_flashinfer_decode_graph_replays_refill_the_captured_plan(lengths, num_heads, page_size):
    """Replays refill the captured wrapper's plan on the GPU; it must match a fresh host plan.

    The graph is captured with short lengths and replayed with long ones, all-ones and long again,
    covering page-table changes, partial last pages and the switch between split and exact plans.
    """
    pytest.importorskip("flashinfer")
    max_len = max(lengths)
    padded = -(-max_len // page_size) * page_size
    batch_size = len(lengths)
    token_map = paged_token_map(batch_size, page_size, padded)
    cache = torch.randn(batch_size * padded, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(1))
    q_nope = torch.randn(batch_size, num_heads, RANK, device="cuda", dtype=torch.bfloat16, generator=gen(2))
    q_rope = torch.randn(batch_size, num_heads, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(3))
    inputs = MLADecodeLayerInputs(q_nope, q_rope, cache)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    builder = FlashInferMLADecodeBackend.get_step_builder_cls()(
        builder_config(num_heads, page_size), workspace=lambda: workspace, max_seq_len=padded, **GRAPH_BUFFERS
    )
    impl = FlashInferMLADecodeBackend.get_impl_cls()(decode_config(num_heads))
    req_ids = list(range(batch_size))

    def graph_batch(step_lengths):
        return make_batch(token_map, req_ids, step_lengths, capacity=padded)

    # The caller rewrites the captured step's b_seq_len in place before each replay.
    captured_batch = graph_batch([2] * batch_size)
    att_state = builder.build(captured_batch)
    graph_out = torch.empty_like(q_nope)
    impl.forward(att_state, inputs, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        impl.forward(att_state, inputs, out=graph_out)

    eager_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    eager_builder = FlashInferMLADecodeBackend.get_step_builder_cls()(
        builder_config(num_heads, page_size), workspace=lambda: eager_workspace, max_seq_len=padded, **GRAPH_BUFFERS
    )
    for replay_lengths in (lengths, [1] * batch_size, lengths):
        replay_batch = graph_batch(replay_lengths)
        assert builder.build(replay_batch).wrapper is att_state.wrapper
        captured_batch.b_seq_len.copy_(replay_batch.b_seq_len)
        graph.replay()
        expected = torch.empty_like(q_nope)
        impl.forward(eager_builder.build(make_batch(token_map, req_ids, replay_lengths)), inputs, out=expected)
        torch.testing.assert_close(graph_out, expected, atol=1e-2, rtol=1e-2)


@pytestmark_cuda
def test_flashinfer_decode_step_builder_keeps_one_graph_wrapper_per_microbatch_and_batch_size():
    pytest.importorskip("flashinfer")
    token_map = paged_token_map(2, 16, 512)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    builder = FlashInferMLADecodeBackend.get_step_builder_cls()(
        builder_config(16, 16), workspace=lambda: workspace, max_seq_len=512, **GRAPH_BUFFERS
    )
    captured = builder.build(make_batch(token_map, [0, 1], [33, 17], capacity=512)).wrapper
    assert builder.build(make_batch(token_map, [0, 1], [300, 257], capacity=512)).wrapper is captured
    assert builder.build(make_batch(token_map, [0, 1], [33, 17], capacity=512, microbatch=1)).wrapper is not captured
    eager = [builder.build(make_batch(token_map, [0, 1], [33, 17])).wrapper for _ in range(2)]
    assert eager[0] is not eager[1] and all(wrapper is not captured for wrapper in eager)


@pytestmark_cuda
def test_flashinfer_decode_shares_one_page_index_buffer_per_microbatch():
    """Every batch that fits, eager or captured at any batch size, uses its microbatch's fixed buffer.

    A buffer per captured batch size would hold batch_size * max_pages entries for each size, which
    for a long context and a large graph batch range costs gigabytes after the KV cache is sized.
    """
    pytest.importorskip("flashinfer")
    token_map = paged_token_map(8, 16, 512)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    builder = FlashInferMLADecodeBackend.get_step_builder_cls()(
        builder_config(16, 16), workspace=lambda: workspace, max_seq_len=512, **GRAPH_BUFFERS
    )
    buffers = [builder.build(make_batch(token_map, [0], [33], microbatch=mb)).wrapper._kv_indices_buf for mb in (0, 1)]
    assert buffers[0].data_ptr() != buffers[1].data_ptr()
    for microbatch in (0, 1):
        for batch_size in (1, 2, 5, 8):
            req_ids = list(range(batch_size))
            for capacity in (None, 512):
                batch = make_batch(token_map, req_ids, [33] * batch_size, capacity=capacity, microbatch=microbatch)
                assert builder.build(batch).wrapper._kv_indices_buf.data_ptr() == buffers[microbatch].data_ptr()
    with pytest.raises(AssertionError, match="CUDA graph batches must fit"):
        builder.build(make_batch(paged_token_map(9, 16, 512), list(range(9)), [33] * 9, capacity=512))


@pytestmark_cuda
def test_fa3_decode_groups_speculative_rows_of_a_request():
    # Two rows per request: the first sees its request's prefix without the last token.
    token_map = paged_token_map(8, 16, 512)
    cache = torch.randn(8 * 512, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(1))
    req_ids, lengths = [2, 2, 0, 0], [40, 41, 299, 300]
    storage = torch.empty(4 * 512, dtype=torch.int32, device="cuda")
    builder = FA3MLADecodeBackend.get_step_builder_cls()(
        page_size=16, page_table_view=lambda rows, width, mb: storage[: rows * -(-width // 16)].view(rows, -1)
    )
    q_nope = torch.randn(4, 16, RANK, device="cuda", dtype=torch.bfloat16, generator=gen(2))
    q_rope = torch.randn(4, 16, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(3))
    out = torch.empty_like(q_nope)
    FA3MLADecodeBackend.get_impl_cls()(decode_config()).forward(
        builder.build(make_batch(token_map, req_ids, lengths, rows=2)),
        MLADecodeLayerInputs(q_nope, q_rope, cache),
        out=out,
    )
    expected = decode_reference(q_nope, q_rope, cache, token_map, req_ids, lengths)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
