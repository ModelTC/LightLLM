from typing import Any
from lightllm.common.basemodel.infer_struct import GraphMode
from unit_tests.common.attention.backends.step_fixture import create_att_state
from lightllm.common.attention.backends.gqa.fa3 import FA3DecodeRuntimeConfig

"""FlashInfer和FA3 MLA后端：数值、CUDA图步骤构建状态和支持的层配置。"""

from unit_tests.common.attention.backends.groups_runner import run_attention

import pytest
import torch

from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.attention.backends.mla import (
    MLADecodeAttConfig,
    MLADecodeAttInputs,
    MLAPrefillAttConfig,
    MLAPrefillAttInputs,
)
from lightllm.common.attention.backends.mla.fa3 import FA3MLADecodeBackend, FA3MLAPrefillBackend
from lightllm.common.attention.backends.mla.flashinfer import (
    FlashInferMLADecodeBackend,
    FlashInferMLAPrefillBackend,
)
from lightllm.common.attention.backends.mla.triton import TritonMLADecodeBackend, TritonMLAPrefillBackend

RANK, ROPE, NOPE, V_DIM, SCALE = 512, 64, 128, 128, 0.1


def prefill_config(heads=16, causal=True):
    return MLAPrefillAttConfig(
        num_heads=heads,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        softmax_scale=SCALE,
        dtype=torch.bfloat16,
        causal=causal,
    )


def decode_config(heads=16, causal=True):
    return MLADecodeAttConfig(
        num_heads=heads,
        kv_lora_rank=RANK,
        qk_rope_head_dim=ROPE,
        softmax_scale=SCALE,
        dtype=torch.bfloat16,
        causal=causal,
    )


def test_non_causal_mla_is_rejected_except_by_fa3():
    for backend in (TritonMLAPrefillBackend, FlashInferMLAPrefillBackend):
        with pytest.raises(NotImplementedError, match="causal"):
            backend(prefill_config(causal=False))
    for backend in (TritonMLADecodeBackend, FlashInferMLADecodeBackend):
        with pytest.raises(NotImplementedError, match="causal"):
            backend(decode_config(causal=False))
    FA3MLAPrefillBackend(prefill_config(causal=False))
    FA3MLADecodeBackend(decode_config(causal=False))


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
    return make_infer_state(
        req_to_token_indexs=token_map,
        b_req_idx=torch.tensor(req_ids, device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        batch_size=len(lengths),
        max_kv_seq_len=max(lengths),
        total_token_num=sum(lengths),
        b1_cu_q_seq_len=cu(q_lens),
        b1_cu_kv_seq_len=cu(lengths),
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
    return MLAPrefillAttInputs(
        q=torch.randn(rows, heads, NOPE + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(2)),
        k_nope=torch.randn(kv_tokens, heads, NOPE, device="cuda", dtype=torch.bfloat16, generator=gen(3)),
        k_rope=torch.randn(kv_tokens, 1, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(4)),
        v=torch.randn(kv_tokens, heads, V_DIM, device="cuda", dtype=torch.bfloat16, generator=gen(5)),
    )


def flashinfer_decode_backend(heads: Any, page_size: Any, workspace: Any, max_seq_len: Any):
    runtime = FlashInferMLADecodeBackend.RuntimeConfig(page_size=page_size, max_seq_len=max_seq_len, max_batch_size=8)
    impl = FlashInferMLADecodeBackend(att_config=decode_config(heads=heads), runtime_config=runtime)
    impl.get_workspace = lambda name, role, numel: workspace
    return impl


def fa3_decode_backend(page_size: Any):
    runtime = FA3DecodeRuntimeConfig(page_size=page_size, max_seq_len=512, max_decode_rows=8)
    return FA3MLADecodeBackend(att_config=decode_config(), runtime_config=runtime)


@pytestmark_cuda
@pytest.mark.parametrize("backend", ["flashinfer", "fa3"])
def test_prefill_with_cached_prefix_matches_reference(backend):
    lengths, ready = [33, 9, 259], [16, 5, 128]
    batch = make_batch(paged_token_map(3, 1, 512), [2, 0, 1], lengths, prefill=True, ready=ready)
    if backend == "flashinfer":
        pytest.importorskip("flashinfer")
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        impl = FlashInferMLAPrefillBackend(prefill_config())
        impl.get_workspace = lambda name, role, numel: workspace
    else:
        impl = FA3MLAPrefillBackend(prefill_config())
    inputs = prefill_inputs(lengths, ready)
    out = torch.empty(inputs.q.shape[0], 16, V_DIM, device="cuda", dtype=torch.bfloat16)
    assert (
        run_attention(
            backend=impl, att_state=create_att_state(backend=impl, infer_state=batch), layer_inputs=inputs, out=out
        )
        is out
    )
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
        impl = flashinfer_decode_backend(heads=16, page_size=page_size, workspace=workspace, max_seq_len=512)
    else:
        impl = fa3_decode_backend(page_size=page_size)
    q_nope = torch.randn(4, 16, RANK, device="cuda", dtype=torch.bfloat16, generator=gen(2))
    q_rope = torch.randn(4, 16, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(3))
    out = torch.empty_like(q_nope)
    run_attention(
        backend=impl,
        att_state=create_att_state(backend=impl, infer_state=batch),
        layer_inputs=MLADecodeAttInputs(q_nope=q_nope, q_rope=q_rope, kv_cache=cache),
        out=out,
    )
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
    inputs = MLADecodeAttInputs(q_nope=q_nope, q_rope=q_rope, kv_cache=cache)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    builder = flashinfer_decode_backend(heads=num_heads, page_size=page_size, workspace=workspace, max_seq_len=padded)
    impl = builder
    req_ids = list(range(batch_size))

    def graph_batch(step_lengths):
        return make_batch(token_map, req_ids, step_lengths, capacity=padded)

    # The caller rewrites the captured step's b_seq_len in place before each replay.
    captured_batch = graph_batch([2] * batch_size)
    att_state = create_att_state(backend=builder, infer_state=captured_batch)
    graph_out = torch.empty_like(q_nope)
    run_attention(backend=impl, att_state=att_state, layer_inputs=inputs, out=graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_attention(backend=impl, att_state=att_state, layer_inputs=inputs, out=graph_out)

    eager_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    eager_builder = flashinfer_decode_backend(
        heads=num_heads, page_size=page_size, workspace=eager_workspace, max_seq_len=padded
    )
    for replay_lengths in (lengths, [1] * batch_size, lengths):
        replay_batch = graph_batch(replay_lengths)
        replay_batch.graph_mode = GraphMode.REPLAY
        assert create_att_state(backend=builder, infer_state=replay_batch).wrapper is None
        captured_batch.copy_for_cuda_graph(new_infer_state=replay_batch)
        graph.replay()
        expected = torch.empty_like(q_nope)
        run_attention(
            backend=impl,
            att_state=create_att_state(
                backend=eager_builder, infer_state=make_batch(token_map, req_ids, replay_lengths)
            ),
            layer_inputs=inputs,
            out=expected,
        )
        torch.testing.assert_close(graph_out, expected, atol=1e-2, rtol=1e-2)


@pytestmark_cuda
def test_decode_wrapper_belongs_to_the_capture_state():
    import weakref

    pytest.importorskip("flashinfer")
    token_map = paged_token_map(2, 16, 512)
    workspace = torch.empty(size=(128 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
    builder = flashinfer_decode_backend(heads=16, page_size=16, workspace=workspace, max_seq_len=512)
    capture_batch = make_batch(token_map, [0, 1], [33, 17], capacity=512)
    captured = create_att_state(backend=builder, infer_state=capture_batch)
    wrapper_ref = weakref.ref(captured.wrapper)
    replay_batch = make_batch(token_map, [0, 1], [300, 257], capacity=512)
    replay_batch.graph_mode = GraphMode.REPLAY
    replay = create_att_state(backend=builder, infer_state=replay_batch)
    assert replay.wrapper is None
    capture_batch.copy_for_cuda_graph(new_infer_state=replay_batch)
    assert captured.wrapper is wrapper_ref()
    del captured, capture_batch
    assert wrapper_ref() is None


@pytestmark_cuda
def test_flashinfer_decode_shares_one_page_index_buffer_per_microbatch():
    """每个适合的批次，无论是急切的还是在任何批次大小下捕获的，都使用其微批次的固定缓冲区。

    每个捕获批次大小的缓冲区将为每个大小保留batch_size * max_pages个条目，
    对于长上下文和大图批次范围，在KV缓存大小后会占用数GB。
    """
    pytest.importorskip("flashinfer")
    token_map = paged_token_map(8, 16, 512)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    builder = flashinfer_decode_backend(heads=16, page_size=16, workspace=workspace, max_seq_len=512)
    buffers = [
        create_att_state(
            backend=builder, infer_state=make_batch(token_map, [0], [33], microbatch=mb)
        ).wrapper._kv_indices_buf
        for mb in (0, 1)
    ]
    assert buffers[0].data_ptr() != buffers[1].data_ptr()
    for microbatch in (0, 1):
        for batch_size in (1, 2, 5, 8):
            req_ids = list(range(batch_size))
            for capacity in (None, 512):
                batch = make_batch(token_map, req_ids, [33] * batch_size, capacity=capacity, microbatch=microbatch)
                assert (
                    create_att_state(backend=builder, infer_state=batch).wrapper._kv_indices_buf.data_ptr()
                    == buffers[microbatch].data_ptr()
                )
    with pytest.raises(AssertionError, match="CUDA graph batches must fit"):
        create_att_state(
            backend=builder, infer_state=make_batch(paged_token_map(9, 16, 512), list(range(9)), [33] * 9, capacity=512)
        )


@pytestmark_cuda
def test_fa3_decode_groups_speculative_rows_of_a_request():
    # Two rows per request: the first sees its request's prefix without the last token.
    token_map = paged_token_map(8, 16, 512)
    cache = torch.randn(8 * 512, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(1))
    req_ids, lengths = [2, 2, 0, 0], [40, 41, 299, 300]
    builder = fa3_decode_backend(page_size=16)
    q_nope = torch.randn(4, 16, RANK, device="cuda", dtype=torch.bfloat16, generator=gen(2))
    q_rope = torch.randn(4, 16, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(3))
    out = torch.empty_like(q_nope)
    builder.decode(
        create_att_state(backend=builder, infer_state=make_batch(token_map, req_ids, lengths, rows=2)),
        MLADecodeAttInputs(q_nope=q_nope, q_rope=q_rope, kv_cache=cache),
        out=out,
    )
    expected = decode_reference(q_nope, q_rope, cache, token_map, req_ids, lengths)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
