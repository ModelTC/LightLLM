from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.windowed_mtp import WindowKVStore, WindowKVPageHelper
from lightllm.common.state_cache_manager.windowed_mtp import WindowedMTPCacheConfig, WindowStateCacheConfig


@pytest.fixture
def tensor_cache():
    from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

    g_cache_manager.cache_env_in()
    try:
        yield g_cache_manager.empty
    finally:
        g_cache_manager.cache_env_out()


def test_checkpoint_and_cpu_rank_layout():
    store = WindowKVStore(3, 2, 1, 3, torch.float32, "cpu", 4)
    expected = torch.arange(48, dtype=torch.float32).view(2, 4, 2, 3)
    store.kv_buffer[:, :4].copy_(expected)
    store.req_to_seq_len[0] = 9
    buffers = SimpleNamespace(kv_buffer=torch.zeros(2, 2, 4, 2, 3), seq_lens=torch.zeros(2, dtype=torch.int64))
    store.save_checkpoint(0, buffers, 1)
    store.restore_checkpoint(2, buffers, 1)
    torch.testing.assert_close(store.kv_buffer[:, 8:12], expected, rtol=0, atol=0)
    assert store.req_to_seq_len.tolist() == [9, 0, 9]

    config = WindowStateCacheConfig(2, 1, 3, 4)
    target = SimpleNamespace(get_cpu_cache_big_page_bytes=lambda: 32)
    layout = WindowedMTPCacheConfig(config, torch.float16, 2, 0, 0, 0, target)
    pages = torch.zeros(2, layout.get_cpu_cache_big_page_bytes(), dtype=torch.uint8)
    for rank in range(2):
        kv, seq_lens = layout.get_window_views(pages, rank)
        kv.fill_(rank + 1)
        seq_lens.fill_(2 ** 35 + rank)
    for rank in range(2):
        kv, seq_lens = layout.get_window_views(pages, rank)
        assert kv.eq(rank + 1).all()
        assert seq_lens.eq(2 ** 35 + rank).all()
    assert not pages[:, :32].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_write_indices_preserve_source_layout(tensor_cache):
    store = WindowKVStore(3, 1, 1, 3, torch.float32, "cuda", 4)
    reqs = torch.tensor([2, 1, 0], dtype=torch.int32, device="cuda")
    q_lengths = torch.tensor([7, 0, 2], dtype=torch.int32, device="cuda")
    ready = torch.tensor([5, 9, 7], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 7, 7], dtype=torch.int32, device="cuda")
    mem_index = store.prepare_write_indices(reqs, starts, ready, q_lengths, token_num=9, alloc_func=tensor_cache)
    # Retain four rows from the long query, skip the empty query, then keep both short-query rows.
    assert mem_index.tolist() == [-1, -1, -1, 8, 9, 10, 11, 3, 0]
    assert store.req_to_seq_len.tolist() == [9, 9, 12]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("window", [1, 4, 8, 257])
def test_commit_and_pack(window, tensor_cache):
    store = WindowKVStore(3, 2, 1, 3, torch.float32, "cuda", window)
    reqs = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    first_len = max(7, window + 3)
    hidden = torch.arange((first_len + 3) * 24, device="cuda", dtype=torch.float32).view(-1, 12)[::2, ::2]
    starts = torch.tensor([0, first_len], dtype=torch.int32, device="cuda")
    first = torch.zeros(2, dtype=torch.int32, device="cuda")
    lengths = torch.tensor([first_len, 3], dtype=torch.int32, device="cuda")
    mem_index = store.prepare_write_indices(
        reqs, starts, first, lengths, token_num=hidden.shape[0], alloc_func=tensor_cache
    )
    for layer in range(2):
        store.write(layer, mem_index, hidden.view(-1, 2, 3) + layer)

    accepted = torch.arange(5 * 6, device="cuda", dtype=torch.float32).view(5, 6) + 1000
    starts = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    first = torch.tensor([first_len, 3], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    mem_index = store.prepare_write_indices(
        reqs, starts, first, lengths, token_num=accepted.shape[0], alloc_func=tensor_cache
    )
    assert mem_index.shape == (5,)
    assert mem_index[[2, 4]].eq(-1).all()
    for layer in range(2):
        store.write(layer, mem_index, accepted.view(-1, 2, 3) + layer)
    history = [torch.cat((hidden[:first_len], accepted[:2])), torch.cat((hidden[first_len:], accepted[3:4]))]
    noise = torch.randn(4, 2, 6, device="cuda")[:, :, ::2]
    for layer in range(2):
        packed = store.pack(layer, reqs, noise, 2, alloc_func=tensor_cache)
        for batch, values in enumerate(history):
            retained = values[-window:].view(-1, 2, 3) + layer
            expected = torch.cat((retained, noise[batch * 2 : (batch + 1) * 2]))
            torch.testing.assert_close(packed[batch, : len(expected)], expected, rtol=0, atol=0)
    assert store.req_to_seq_len.tolist() == [first_len + 2, 0, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sliding_window", [0, 4])
def test_fa3_graph_refreshes_request_state(tensor_cache, sliding_window):
    from lightllm.common.basemodel.attention.fa3.windowed_mtp import WindowedMTPDecodeAttState
    from lightllm.common.basemodel.cuda_graph import CudaGraph
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
    from lightllm.models.qwen3_dflash2.layer_infer.transformer_layer_infer import Qwen3DFlash2TransformerLayerInfer
    from lightllm.utils.sgl_utils import flash_attn_with_kvcache

    if flash_attn_with_kvcache is None:
        pytest.skip("FA3 required")
    torch.manual_seed(0)
    window, block_size, head_dim = 4, 2, 64
    store = WindowKVStore(3, 1, 1, head_dim, torch.bfloat16, "cuda", window)
    store.kv_buffer.normal_()
    store.req_to_seq_len.copy_(torch.tensor([7, 0, 2], device="cuda"))
    backend = SimpleNamespace(
        model=SimpleNamespace(block_size=block_size, draft_layer_start=0, uses_windowed_draft_kv=True)
    )

    def make_state(req_ids):
        state = InferStateInfo()
        state.is_prefill = False
        state.input_ids = torch.zeros(4, dtype=torch.int64, device="cuda")
        state.b_req_idx = torch.tensor(req_ids, dtype=torch.int32, device="cuda").repeat_interleave(block_size)
        state.mem_manager = SimpleNamespace(windowed_draft_kv=store)
        state.query = torch.randn(4, 2, head_dim, dtype=torch.bfloat16, device="cuda")
        state.noise_kv = torch.randn(4, 2, head_dim, dtype=torch.bfloat16, device="cuda")
        state.decode_att_state = WindowedMTPDecodeAttState(backend=backend, infer_state=state)
        state.init_att_state()
        return state

    layer_class = Qwen3DFlash2TransformerLayerInfer if sliding_window else Qwen3DFlashTransformerLayerInfer
    layer = layer_class.__new__(layer_class)
    layer.sliding_window_ = sliding_window
    layer.layer_num_, layer.tp_q_head_num_, layer.tp_k_head_num_, layer.head_dim_ = 0, 2, 1, head_dim
    layer._get_qkv = lambda input, state, weight: (state.query, state.noise_kv)
    layer._get_o = lambda output, state, weight: output.flatten(1)
    weight = SimpleNamespace()
    captured = make_state([0, 2])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer.token_attention_forward(captured.query.flatten(1), captured, weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer.token_attention_forward(captured.query.flatten(1), captured, weight)
    runner = CudaGraph.__new__(CudaGraph)
    runner.graph = {4: (graph, captured, output)}

    for req_ids, lengths in [([2, 1], [7, 6, 0]), ([1, 0], [2, 1, 0])]:
        store.kv_buffer.normal_()
        store.req_to_seq_len.copy_(torch.tensor(lengths, device="cuda"))
        runtime = make_state(req_ids)
        att_state = runtime.decode_att_state
        expected_lengths = [min(lengths[req], window) + block_size for req in req_ids]
        assert att_state.cache_seqlens.tolist() == expected_lengths
        expected = []
        for batch, req in enumerate(req_ids):
            positions = torch.arange(max(0, lengths[req] - window), lengths[req], device="cuda")
            history = store.kv_buffer[0, req * window + positions % window]
            noise = runtime.noise_kv[batch * block_size : (batch + 1) * block_size]
            kv = torch.cat((history, noise)).float()
            q = runtime.query[batch * block_size : (batch + 1) * block_size].float()
            scores = torch.einsum("qhd,kd->hqk", q, kv[:, 0]) * head_dim**-0.5
            if sliding_window:
                q_pos = torch.arange(block_size, device="cuda") + len(history)
                k_pos = torch.arange(len(kv), device="cuda")
                mask = (q_pos[:, None] - k_pos[None, :]).abs() >= sliding_window
                scores.masked_fill_(mask[None], float("-inf"))
            expected.append(torch.einsum("hqk,kd->qhd", scores.softmax(-1), kv[:, 1]))
        expected = torch.cat(expected).flatten(1).to(runtime.query.dtype)
        actual = runner._replay(runtime)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefill_tp,decode_tp", [(1, 1), (1, 2), (2, 1)])
def test_pd_page_round_trip(monkeypatch, prefill_tp, decode_tp):
    def make_mems(tp):
        monkeypatch.setattr("lightllm.utils.envs_utils.get_env_start_args", lambda: SimpleNamespace(tp=tp, dp=1))
        mems = [
            SimpleNamespace(
                windowed_draft_kv=WindowKVStore(3, 2, 2 // tp, 8, torch.float32, "cuda", 4),
                kv_move_buffer=torch.zeros(1, 2048, dtype=torch.uint8, device="cuda"),
            )
            for _ in range(tp)
        ]
        helper = WindowKVPageHelper(mems[0], offset=16)
        helper.assert_page_size()
        return mems, helper

    source, writer = make_mems(prefill_tp)
    destination, reader = make_mems(decode_tp)
    global_kv = torch.randn(2, 4, 4, 8, device="cuda")

    def shard_heads(rank, tp):
        start = rank * (2 // tp)
        return list(range(start, start + 2 // tp)) + list(range(2 + start, 2 + start + 2 // tp))

    for rank, mem in enumerate(source):
        mem.windowed_draft_kv.kv_buffer[:, :4].copy_(global_kv[:, :, shard_heads(rank, prefill_tp)])
        mem.windowed_draft_kv.req_to_seq_len[0] = 7
    source[0].kv_move_buffer[:, :16].fill_(42)
    writer.copy_req_page(0, 0, source, "write")
    destination[0].kv_move_buffer.copy_(source[0].kv_move_buffer)
    for mem in destination:
        mem.windowed_draft_kv.kv_buffer.zero_()
    reader.copy_req_page(0, 2, destination, "read")
    for rank, mem in enumerate(destination):
        store = mem.windowed_draft_kv
        torch.testing.assert_close(store.kv_buffer[:, 8:12], global_kv[:, :, shard_heads(rank, decode_tp)])
        assert store.req_to_seq_len.tolist() == [0, 0, 7]
        assert not store.kv_buffer[:, :8].any()
    assert destination[0].kv_move_buffer[:, :16].eq(42).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tail_checkpoint_copy_preserves_stream_order():
    from lightllm.common.kv_cache_mem_manager.operator.windowed_mtp import WindowedMTPMemOperator
    from lightllm.common.state_cache_manager.windowed_mtp import WindowStateBuffers

    config = WindowStateCacheConfig(2, 1, 8, 4)
    source = SimpleNamespace(draft_window=WindowStateBuffers(2, config, torch.bfloat16))
    destination = SimpleNamespace(draft_window=WindowStateBuffers(2, config, torch.bfloat16))
    operator = WindowedMTPMemOperator.__new__(WindowedMTPMemOperator)
    operator.mem_manager = SimpleNamespace(big_page_buffers=destination)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000)
        expected = torch.randn(source.draft_window.kv_buffer[1].shape, dtype=torch.bfloat16, device="cuda")
        seq_len = torch.tensor([2 ** 35 + 7], dtype=torch.int64, device="cuda")
        source.draft_window.kv_buffer[1].copy_(expected, non_blocking=True)
        source.draft_window.seq_lens[1:2].copy_(seq_len, non_blocking=True)
        operator._copy_tail_state(source, source_id=1, destination_id=0)
    stream.synchronize()
    torch.testing.assert_close(destination.draft_window.kv_buffer[0], expected.cpu(), rtol=0, atol=0)
    assert destination.draft_window.seq_lens.tolist() == [2 ** 35 + 7, 0]
    assert not destination.draft_window.kv_buffer[1].any()
