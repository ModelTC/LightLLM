import weakref
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

from lightllm.common.attention.backend import AttentionMetadata, TensorAllocator
from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.triton import TritonGQADecodeBackend, TritonGQAPrefillBackend
from lightllm.common.basemodel.attention_runtime import AttentionStepConfig, build_attention_metadata
from lightllm.common.basemodel.infer_struct import GraphMode, InferStateInfo


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(params=["torch", "serving"])
def allocator(request: pytest.FixtureRequest) -> Iterator[TensorAllocator]:
    if request.param == "torch":
        yield torch.empty
        return

    from lightllm.common.basemodel.layer_infer.base_layer_infer import BaseLayerInfer
    from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

    g_cache_manager.cache_env_in()
    try:
        yield BaseLayerInfer.__new__(BaseLayerInfer).alloc_tensor
    finally:
        g_cache_manager.cache_env_out()


def reference(q, k, v, token_map, req_ids, lengths, ready, window):
    outputs = []
    offset = 0
    for req_id, length, cached in zip(req_ids.tolist(), lengths.tolist(), ready):
        count = length - cached
        indices = token_map[req_id, :length].long()
        keys = k[indices].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        values = v[indices].float().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[offset : offset + count].float(), keys) * q.shape[-1] ** -0.5
        query_pos = torch.arange(cached, length, device=q.device)[:, None]
        key_pos = torch.arange(length, device=q.device)[None, :]
        mask = key_pos <= query_pos
        if window != (-1, -1):
            mask &= key_pos >= query_pos - window[0]
        scores.masked_fill_(~mask[None, :, :], float("-inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        offset += count
    return torch.cat(outputs)


def make_batch(token_map, req_ids, lengths, **kwargs) -> AttentionMetadata:
    return AttentionMetadata(
        req_to_token_indexs=token_map,
        b_req_idx=req_ids,
        b_seq_len=lengths,
        batch_size=lengths.shape[0],
        max_kv_seq_len=int(lengths.max()),
        total_token_num=int(lengths.sum()),
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads,window", [(4, 4, (-1, -1)), (8, 2, (-1, -1)), (8, 2, (15, 0))])
@pytest.mark.parametrize("cached", [False, True])
def test_prefill_decode(dtype, q_heads, kv_heads, window, cached):
    torch.manual_seed(42)
    lengths = torch.tensor([33, 9, 259], device="cuda", dtype=torch.int32)
    ready = [16, 5, 128] if cached else [0, 0, 0]
    token_map = torch.randperm(3 * 272, device="cuda", dtype=torch.int32).view(3, 272)
    req_ids = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    kv = torch.randn(3 * 272, 2 * kv_heads, 64, device="cuda", dtype=dtype)
    before = kv.clone()
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]
    query_lengths = lengths - torch.tensor(ready, device="cuda", dtype=torch.int32)
    q = torch.randn(int(query_lengths.sum()), q_heads + 1, 64, device="cuda", dtype=dtype)[:, :q_heads]
    q_before = q.clone()
    layer_config = GQALayerConfig(
        num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=dtype, sliding_window=window
    )
    att_state = TritonGQAPrefillBackend.build(
        make_batch(
            token_map,
            req_ids,
            lengths,
            b_q_start_loc=query_lengths.cumsum(0, dtype=torch.int32) - query_lengths,
            b_ready_cache_len=torch.tensor(ready, device="cuda", dtype=torch.int32),
            max_q_seq_len=int(query_lengths.max()),
        ),
    )
    out = torch.empty(q.shape[0], q_heads + 1, 64, device=q.device, dtype=q.dtype)[:, :q_heads]
    impl = TritonGQAPrefillBackend(layer_config)
    assert impl.forward(att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v), out=out) is out
    torch.testing.assert_close(
        out.float(), reference(q, k, v, token_map, req_ids, lengths, ready, window), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(q, q_before, atol=0, rtol=0)

    def allocate(shape: tuple[int, ...] | list[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        assert device == q.device
        return torch.empty(shape, dtype=dtype, device=device)

    for _ in range(3):
        lengths += 1
        q = torch.randn(3, q_heads + 1, 64, device="cuda", dtype=dtype)[:, :q_heads]
        q_before = q.clone()
        att_state = TritonGQADecodeBackend.build(
            make_batch(token_map, req_ids, lengths), TritonGQADecodeBackend.build_context_type()
        )
        out = torch.empty(3, q_heads + 1, 64, device=q.device, dtype=q.dtype)[:, :q_heads]
        impl = TritonGQADecodeBackend(layer_config)
        assert impl.forward(att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v), out=out, alloc_func=allocate) is out
        torch.testing.assert_close(
            out.float(),
            reference(q, k, v, token_map, req_ids, lengths, (lengths - 1).tolist(), window),
            atol=0.02,
            rtol=0.02,
        )
        torch.testing.assert_close(q, q_before, atol=0, rtol=0)
    torch.testing.assert_close(kv, before, atol=0, rtol=0)


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
def test_decode_graph_built_from_the_step(q_heads, kv_heads, allocator):
    torch.manual_seed(43)
    owner = InferStateInfo()
    owner.batch_size = 2
    owner.b_req_idx = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    owner.b_seq_len = torch.tensor([33, 17], device="cuda", dtype=torch.int32)
    owner.max_kv_seq_len = 33
    token_map = torch.randperm(2 * 512, device="cuda", dtype=torch.int32).view(2, 512)
    owner.req_manager = SimpleNamespace(req_to_token_indexs=token_map)
    step_config = AttentionStepConfig(cuda_graph_max_kv_seq_len=512)
    layer_config = GQALayerConfig(num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=torch.bfloat16)
    impl = TritonGQADecodeBackend(layer_config)
    # As in the serving decode path, the capture flag is set before the attention state are built and
    # capture raises max_kv_seq_len to the graph capacity only afterwards.
    owner.graph_mode = GraphMode.CAPTURE
    att_state = TritonGQADecodeBackend.build(
        build_attention_metadata(owner, step_config), TritonGQADecodeBackend.create_build_context(layer_config, None)
    )
    owner.max_kv_seq_len = 512
    q = torch.randn(2, q_heads, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1024, 2 * kv_heads, 64, device="cuda", dtype=q.dtype)
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]

    def forward() -> torch.Tensor:
        out = allocator(q.shape, dtype=q.dtype, device=q.device)
        return impl.forward(att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v), out=out, alloc_func=allocator)

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    for lengths in ([129, 65], [300, 257]):
        owner.b_seq_len.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        q.normal_()
        graph.replay()
        expected = reference(q, k, v, token_map, owner.b_req_idx, owner.b_seq_len, [x - 1 for x in lengths], (-1, -1))
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
    # The attention state keep the capture step's actual length as the config hint and the capacity as the bound.
    assert (att_state.max_kv_seq_len, att_state.kv_seq_len_bound) == (33, 512)
    # Attention state do not keep the step's InferStateInfo alive.
    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None


def test_list_sliding_window_matches_tuple():
    # Model configs loaded from JSON provide windows as lists.
    layer_config = GQALayerConfig(
        num_heads=4, num_kv_heads=4, head_dim=64, dtype=torch.bfloat16, sliding_window=[-1, -1]
    )
    assert layer_config.sliding_window == (-1, -1)
    q = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(64, 8, 64, device="cuda", dtype=q.dtype)
    token_map = torch.arange(64, device="cuda", dtype=torch.int32).view(2, 32)
    lengths = torch.tensor([5, 9], device="cuda", dtype=torch.int32)
    req_ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    att_state = TritonGQADecodeBackend.build(
        make_batch(token_map, req_ids, lengths), TritonGQADecodeBackend.build_context_type()
    )
    impl = TritonGQADecodeBackend(layer_config)
    out = impl.forward(att_state, GQALayerInputs(q=q, k_cache=kv[:, :4], v_cache=kv[:, 4:]), out=torch.empty_like(q))
    expected = reference(q, kv[:, :4], kv[:, 4:], token_map, req_ids, lengths, [4, 8], (-1, -1))
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("q_heads,kv_heads", [(2, 4), (6, 4)])
def test_spec_rejects_heads_not_multiple_of_kv_heads(q_heads, kv_heads):
    with pytest.raises(
        ValueError, match=rf"num_heads \({q_heads}\) must be a positive multiple of num_kv_heads \({kv_heads}\)"
    ):
        GQALayerConfig(num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=torch.bfloat16)


def test_mha_decode_rejects_sliding_window_at_construction():
    layer_config = GQALayerConfig(
        num_heads=4, num_kv_heads=4, head_dim=64, dtype=torch.bfloat16, sliding_window=(15, 0)
    )
    with pytest.raises(NotImplementedError, match="MHA decode"):
        TritonGQADecodeBackend(layer_config)


def test_graph_decode_bounds_lengths_by_capacity():
    lengths = torch.tensor([33, 17], device="cuda", dtype=torch.int32)
    req_ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    token_map = torch.zeros(2, 512, device="cuda", dtype=torch.int32)
    step_builder = TritonGQADecodeBackend.build_context_type()
    eager = TritonGQADecodeBackend.build(make_batch(token_map, req_ids, lengths), step_builder)
    graph = TritonGQADecodeBackend.build(
        make_batch(token_map, req_ids, lengths, cuda_graph_max_kv_seq_len=512), step_builder
    )
    assert (eager.kv_seq_len_bound, eager.max_kv_seq_len) == (33, 33)
    assert (graph.kv_seq_len_bound, graph.max_kv_seq_len) == (512, 33)


@pytest.mark.parametrize("window", [(-1, 0), (0, -1), (-2, -2)])
def test_spec_rejects_malformed_window(window):
    with pytest.raises(ValueError, match="sliding_window"):
        GQALayerConfig(num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, sliding_window=window)


def test_spec_defaults_scale_and_triton_rejects_other_scales_and_right_windows():
    layer_config = GQALayerConfig(num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16)
    assert layer_config.softmax_scale == 0.125 and layer_config.has_default_scale
    for backend in (TritonGQAPrefillBackend, TritonGQADecodeBackend):
        backend(layer_config)
        with pytest.raises(NotImplementedError, match="softmax_scale"):
            backend(GQALayerConfig(num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, softmax_scale=1.0))
        # A two-sided window is a valid layer_config (non-causal block drafting) but not a Triton GQA mode.
        two_sided = GQALayerConfig(
            num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, sliding_window=(7, 7), causal=False
        )
        with pytest.raises(NotImplementedError, match="causal sliding windows"):
            backend(two_sided)


def test_spec_rejects_causal_window_that_looks_right():
    with pytest.raises(ValueError, match="causal sliding_window"):
        GQALayerConfig(num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, sliding_window=(7, 7))
    GQALayerConfig(num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, sliding_window=(7, 0))


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("fields", [{"causal": False}, {"has_sinks": True}])
def test_layers_triton_cannot_run_are_rejected_when_selected(phase, fields):
    # Rather than silently running causal attention without sinks.
    from lightllm.common.attention import selector

    layer_config = GQALayerConfig(num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, **fields)
    with pytest.raises(ValueError, match="causal attention without sinks"):
        selector.select(layer_config, phase, "triton")


# The shared-group kernel has no window, so windowed layers must take the per-row kernel.
@pytest.mark.parametrize("q_heads,kv_heads,window", [(8, 8, (-1, -1)), (8, 2, (-1, -1)), (8, 2, (15, 0))])
def test_speculative_decode_rows_share_request_kv(q_heads, kv_heads, window):
    # Two rows per request; the last row of each sees the longest prefix. PAD rows are padding.
    pad = 7
    token_map = torch.randperm(8 * 512, device="cuda", dtype=torch.int32).view(8, 512)
    req_ids = torch.tensor([2, 2, 0, 0, pad, pad], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([40, 41, 299, 300, 1, 1], device="cuda", dtype=torch.int32)
    kv = torch.randn(8 * 512, 2 * kv_heads, 64, device="cuda", dtype=torch.bfloat16)
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]
    q = torch.randn(6, q_heads, 64, device="cuda", dtype=torch.bfloat16)
    batch = make_batch(token_map, req_ids, lengths, cu_kv_seq_len=None, decode_rows_per_request=2, pad_req_idx=pad)
    att_state = TritonGQADecodeBackend.build(batch, TritonGQADecodeBackend.build_context_type())
    assert att_state.b_mark_shared_group.cpu().tolist() == [0, 2, 0, 2, 1, 1]
    layer_config = GQALayerConfig(
        num_heads=q_heads, num_kv_heads=kv_heads, head_dim=64, dtype=torch.bfloat16, sliding_window=window
    )
    out = torch.empty_like(q)
    TritonGQADecodeBackend(layer_config).forward(att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v), out=out)
    expected = reference(q, k, v, token_map, req_ids, lengths, (lengths - 1).tolist(), window)
    torch.testing.assert_close(out[:4].float(), expected[:4], atol=0.02, rtol=0.02)


def test_speculative_decode_groups_are_kept_per_microbatch_and_batch_size():
    token_map = torch.zeros(8, 512, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([5, 6, 5, 6], device="cuda", dtype=torch.int32)
    step_builder = TritonGQADecodeBackend.build_context_type()

    def build(req_ids, **kwargs):
        req_ids = torch.tensor(req_ids, device="cuda", dtype=torch.int32)
        batch = make_batch(token_map, req_ids, lengths, decode_rows_per_request=2, pad_req_idx=7, **kwargs)
        return TritonGQADecodeBackend.build(batch, step_builder)

    captured = build([1, 1, 2, 2], cuda_graph_max_kv_seq_len=512)
    assert captured.b_mark_shared_group.cpu().tolist() == [0, 2, 0, 2]
    # A replay rewrites the captured groups in place, so the graph reads the new layout.
    replayed = build([3, 4, 4, 7], cuda_graph_max_kv_seq_len=512)
    assert replayed.b_mark_shared_group is captured.b_mark_shared_group
    assert captured.b_mark_shared_group.cpu().tolist() == [1, 0, 2, 1]
    other = build([1, 1, 2, 2], cuda_graph_max_kv_seq_len=512, microbatch_index=1)
    assert other.b_mark_shared_group is not captured.b_mark_shared_group
    assert build([1, 1, 2, 2]).b_mark_shared_group is not captured.b_mark_shared_group


def alibi_reference(q, k, v, token_map, req_ids, lengths, ready, slopes):
    outputs, offset = [], 0
    for req_id, length, cached in zip(req_ids.tolist(), lengths.tolist(), ready):
        count = length - cached
        indices = token_map[req_id, :length].long()
        scores = torch.einsum("qhd,khd->hqk", q[offset : offset + count].float(), k[indices].float()) * 64**-0.5
        query_pos = torch.arange(cached, length, device=q.device)[:, None]
        key_pos = torch.arange(length, device=q.device)[None, :]
        scores = scores + slopes.float()[:, None, None] * (key_pos - query_pos)[None]
        scores.masked_fill_(~(key_pos <= query_pos)[None], float("-inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[indices].float()))
        offset += count
    return torch.cat(outputs)


def test_alibi_prefill_and_decode_match_reference():
    heads = 4
    token_map = torch.randperm(3 * 512, device="cuda", dtype=torch.int32).view(3, 512)
    req_ids = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([33, 9, 259], device="cuda", dtype=torch.int32)
    ready = [16, 5, 128]
    kv = torch.randn(3 * 512, 2 * heads, 64, device="cuda", dtype=torch.bfloat16)
    k, v = kv[:, :heads], kv[:, heads:]
    slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625], device="cuda")
    layer_config = GQALayerConfig(
        num_heads=heads, num_kv_heads=heads, head_dim=64, dtype=torch.bfloat16, has_alibi=True
    )
    q_lens = [n - r for n, r in zip(lengths.tolist(), ready)]
    q = torch.randn(sum(q_lens), heads, 64, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, *q_lens], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    batch = make_batch(
        token_map,
        req_ids,
        lengths,
        b_q_start_loc=cu_q[:-1],
        b_ready_cache_len=torch.tensor(ready, device="cuda", dtype=torch.int32),
        max_q_seq_len=max(q_lens),
    )
    out = torch.empty_like(q)
    TritonGQAPrefillBackend(layer_config).forward(
        TritonGQAPrefillBackend.build(batch),
        GQALayerInputs(q=q, k_cache=k, v_cache=v, alibi_slopes=slopes),
        out=out,
    )
    expected = alibi_reference(q, k, v, token_map, req_ids, lengths, ready, slopes)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)

    q = torch.randn(3, heads, 64, device="cuda", dtype=torch.bfloat16)
    cu_kv = torch.tensor([0, *lengths.tolist()], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    att_state = TritonGQADecodeBackend.build(
        make_batch(token_map, req_ids, lengths, cu_kv_seq_len=cu_kv), TritonGQADecodeBackend.build_context_type()
    )
    out = torch.empty_like(q)
    TritonGQADecodeBackend(layer_config).forward(
        att_state, GQALayerInputs(q=q, k_cache=k, v_cache=v, alibi_slopes=slopes), out=out
    )
    expected = alibi_reference(q, k, v, token_map, req_ids, lengths, (lengths - 1).tolist(), slopes)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)


def test_alibi_is_supported_only_for_triton_mha_without_window():
    from lightllm.common.attention.backends.gqa.fa3 import FA3GQAPrefillBackend
    from lightllm.common.attention.backends.gqa.flashinfer import FlashInferGQAPrefillBackend

    def layer_config(**fields):
        fields = {
            "num_heads": 4,
            "num_kv_heads": 4,
            "head_dim": 64,
            "dtype": torch.bfloat16,
            "has_alibi": True,
            **fields,
        }
        return GQALayerConfig(**fields)

    for fields in (dict(num_kv_heads=2), dict(sliding_window=(15, 0))):
        with pytest.raises(NotImplementedError, match="ALiBi"):
            TritonGQAPrefillBackend(layer_config(**fields))
    for backend in (FA3GQAPrefillBackend, FlashInferGQAPrefillBackend):
        with pytest.raises(NotImplementedError):
            backend(layer_config())


@pytest.mark.parametrize(
    "window,head_dim,spans",
    [
        ((-1, -1), 128, {0: (4, 30), 2: (100, 160)}),
        ((15, 0), 256, {}),
        # Spans longer than the window, one starting inside the cached prefix: image queries still
        # see their whole span, and keys outside both the window and the span stay hidden.
        ((15, 0), 256, {0: (4, 30), 2: (100, 160)}),
        ((15, 0), 128, {0: (4, 30), 2: (100, 160)}),
    ],
)
def test_bidirectional_image_spans_prefill_matches_reference(window, head_dim, spans):
    q_heads, kv_heads = 8, 2
    token_map = torch.randperm(3 * 512, device="cuda", dtype=torch.int32).view(3, 512)
    req_ids = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([33, 9, 259], device="cuda", dtype=torch.int32)
    ready = [0, 5, 128]
    # spans: request index -> image positions [start, end)
    q_lens = [n - r for n, r in zip(lengths.tolist(), ready)]
    image_start, image_end = (
        torch.tensor(
            [
                spans[i][side] if i in spans and spans[i][0] <= pos < spans[i][1] else 0
                for i, (n, r) in enumerate(zip(lengths.tolist(), ready))
                for pos in range(r, n)
            ],
            device="cuda",
            dtype=torch.int32,
        )
        for side in (0, 1)
    )
    kv = torch.randn(3 * 512, 2 * kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k, v = kv[:, :kv_heads], kv[:, kv_heads:]
    q = torch.randn(sum(q_lens), q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, *q_lens], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    batch = make_batch(
        token_map,
        req_ids,
        lengths,
        b_q_start_loc=cu_q[:-1],
        b_ready_cache_len=torch.tensor(ready, device="cuda", dtype=torch.int32),
        max_q_seq_len=max(q_lens),
        b_image_token_end=image_end,
        b_image_token_start=image_start,
    )
    layer_config = GQALayerConfig(
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        sliding_window=window,
        bidirectional_image_spans=True,
    )
    out = torch.empty_like(q)
    TritonGQAPrefillBackend(layer_config).forward(
        TritonGQAPrefillBackend.build(batch),
        GQALayerInputs(q=q, k_cache=k, v_cache=v),
        out=out,
    )

    expected, row = [], 0
    for req_id, length, cached in zip(req_ids.tolist(), lengths.tolist(), ready):
        indices = token_map[req_id, :length].long()
        keys = k[indices].float().repeat_interleave(q_heads // kv_heads, dim=1)
        values = v[indices].float().repeat_interleave(q_heads // kv_heads, dim=1)
        count = length - cached
        query_pos = torch.arange(cached, length, device="cuda")[:, None]
        key_pos = torch.arange(length, device="cuda")[None, :]
        mask = key_pos <= query_pos
        if window != (-1, -1):
            mask &= key_pos >= query_pos - window[0]
        # Image queries also see their whole span.
        mask |= (key_pos >= image_start[row : row + count, None]) & (key_pos < image_end[row : row + count, None])
        scores = torch.einsum("qhd,khd->hqk", q[row : row + count].float(), keys) * head_dim**-0.5
        scores.masked_fill_(~mask[None], float("-inf"))
        expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        row += count
    torch.testing.assert_close(out.float(), torch.cat(expected), atol=0.02, rtol=0.02)


def test_bidirectional_image_spans_are_prefill_only():
    layer_config = GQALayerConfig(
        num_heads=8, num_kv_heads=2, head_dim=64, dtype=torch.bfloat16, bidirectional_image_spans=True
    )
    TritonGQAPrefillBackend(layer_config)
    with pytest.raises(NotImplementedError, match="prefill only"):
        TritonGQADecodeBackend(layer_config)


def test_head_dim_limit_of_alibi_and_unwindowed_image_span_kernels_is_checked_at_construction():
    # Those kernels stop at head_dim 256; the windowed image-span and plain kernels go to 512.
    def layer_config(**fields):
        return GQALayerConfig(**{"num_heads": 8, "num_kv_heads": 8, "dtype": torch.bfloat16, **fields})

    for fields in (dict(has_alibi=True), dict(num_kv_heads=2, bidirectional_image_spans=True)):
        TritonGQAPrefillBackend(layer_config(head_dim=256, **fields))
        with pytest.raises(NotImplementedError, match="head_dim <= 256"):
            TritonGQAPrefillBackend(layer_config(head_dim=512, **fields))
    TritonGQAPrefillBackend(
        layer_config(head_dim=512, num_kv_heads=2, bidirectional_image_spans=True, sliding_window=(1023, 0))
    )
    TritonGQAPrefillBackend(layer_config(head_dim=512, num_kv_heads=2))
