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
@pytest.mark.parametrize("window", [1, 4, 8])
def test_commit_pack_and_graph_replay(window, tensor_cache):
    store = WindowKVStore(3, 2, 1, 3, torch.float32, "cuda", window)
    reqs = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    hidden = torch.arange(20 * 12, device="cuda", dtype=torch.float32).view(20, 12)[::2, ::2]
    starts = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
    first = torch.zeros(2, dtype=torch.int32, device="cuda")
    lengths = torch.tensor([7, 3], dtype=torch.int32, device="cuda")
    features, _, mem_index = store.prepare(reqs, hidden, starts, first, lengths, 7, alloc_func=tensor_cache)
    for layer in range(2):
        store.write(layer, mem_index, features.view(-1, 2, 3) + layer)

    accepted = torch.arange(5 * 6, device="cuda", dtype=torch.float32).view(5, 6) + 1000
    starts = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    first = torch.tensor([7, 3], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    features, _, mem_index = store.prepare(reqs, accepted, starts, first, lengths, 3, alloc_func=tensor_cache)
    for layer in range(2):
        store.write(layer, mem_index, features.view(-1, 2, 3) + layer)
    history = [torch.cat((hidden[:7], accepted[:2])), torch.cat((hidden[7:10], accepted[3:4]))]
    noise = torch.randn(4, 2, 6, device="cuda")[:, :, ::2]
    for layer in range(2):
        packed = store.pack(layer, reqs, noise, 2, alloc_func=tensor_cache)
        expected = torch.zeros_like(packed)
        for batch, values in enumerate(history):
            retained = values[-window:].view(-1, 2, 3) + layer
            expected[batch, : len(retained)] = retained
            expected[batch, len(retained) : len(retained) + 2] = noise[batch * 2 : (batch + 1) * 2]
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    assert store.req_to_seq_len.tolist() == [9, 0, 4]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        store.pack(1, reqs, noise, 2, alloc_func=tensor_cache)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = store.pack(1, reqs, noise, 2, alloc_func=tensor_cache)
    reqs.fill_(1)
    graph.replay()
    expected.zero_()
    expected[:, :2] = noise.reshape(2, 2, 2, 3)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    store.kv_buffer[:, window : window + 1].fill_(42)
    store.req_to_seq_len[1] = 1
    graph.replay()
    expected.zero_()
    expected[:, 0] = 42
    expected[:, 1:3] = noise.reshape(2, 2, 2, 3)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tp_world_size", [1, 2, 4])
def test_cpu_transfer_partial_pages(monkeypatch, tp_world_size):
    from lightllm.common.basemodel.triton_kernel.kv_cache_offload import load_cpu_kv_to_gpu, offload_gpu_kv_to_cpu

    args = SimpleNamespace(cpu_cache_token_page_size=4, linear_att_hash_page_size=2, linear_att_page_block_num=2)
    monkeypatch.setattr("lightllm.utils.envs_utils.get_env_start_args", lambda: args)
    config = WindowedMTPCacheConfig(WindowStateCacheConfig(1, 1, 8, 4), torch.float32, tp_world_size, 2, 8, 2)
    pages = torch.zeros((1, config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8, pin_memory=True)
    cpu_kv = config.get_full_att_kv_view(pages)
    replicas = max(1, tp_world_size // 2)
    kvs = [torch.randn(2, 8, 2 * max(1, 2 // tp_world_size), 8, device="cuda") for _ in range(tp_world_size)]
    for rank in range(tp_world_size):
        kvs[rank].copy_(kvs[rank // replicas * replicas])
    expected = [kv.clone() for kv in kvs]
    mem_index = torch.tensor([1, 5, -1, -1], dtype=torch.int32, device="cuda")
    page_index = torch.zeros(1, dtype=torch.int64, device="cuda")
    readies = torch.zeros(1, dtype=torch.bool, device="cuda")
    for rank, kv in enumerate(kvs):
        offload_gpu_kv_to_cpu(mem_index, kv, None, cpu_kv, None, page_index, readies, rank, tp_world_size, 16)
    for rank, kv in enumerate(kvs):
        kv.zero_()
        load_cpu_kv_to_gpu(mem_index, kv, None, cpu_kv, None, page_index, rank, tp_world_size, 16)
        torch.testing.assert_close(kv[:, [1, 5]], expected[rank][:, [1, 5]], rtol=0, atol=0)
        assert not kv[:, [0, 2, 3, 4, 6, 7]].any()
    assert not pages[:, config.get_cpu_cache_window_offset() :].any()


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
