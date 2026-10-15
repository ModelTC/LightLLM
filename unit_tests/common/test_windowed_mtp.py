from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.windowed_mtp import WindowKVStore
from lightllm.common.kv_cache_mem_manager.windowed_mtp import WindowKVPageHelper
from lightllm.common.state_cache_manager.windowed_mtp import (
    WindowedMTPCacheConfig,
    WindowStateBuffers,
    WindowStateCacheConfig,
)


def _checkpoint_buffers(slots, layers, window, kv_heads, head_dim, dtype):
    return SimpleNamespace(
        kv_buffer=torch.zeros((slots, layers, window, 2 * kv_heads, head_dim), dtype=dtype),
        seq_lens=torch.zeros(slots, dtype=torch.int64),
    )


@pytest.fixture
def tensor_cache():
    from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

    g_cache_manager.cache_env_in()
    try:
        yield g_cache_manager.empty
    finally:
        g_cache_manager.cache_env_out()


def test_checkpoint_round_trip_preserves_physical_ring_layout():
    layers = 2
    window = 4
    kv_heads = 2
    head_dim = 3
    source_req = 1
    destination_req = 2
    checkpoint_slot = 1

    store = WindowKVStore(
        requests=3,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
        window=window,
    )
    expected_kv = torch.arange(
        layers * window * 2 * kv_heads * head_dim,
        dtype=torch.float32,
    ).view(layers, window, 2 * kv_heads, head_dim)
    store.kv_buffer[:, source_req * window : (source_req + 1) * window].copy_(expected_kv)
    store.req_to_seq_len[source_req] = 9

    buffers = _checkpoint_buffers(
        slots=2,
        layers=layers,
        window=window,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=store.kv_buffer.dtype,
    )
    store.save_checkpoint(source_req, buffers, checkpoint_slot)

    store.kv_buffer[:, destination_req * window : (destination_req + 1) * window].fill_(-1)
    store.req_to_seq_len[destination_req] = -1
    store.restore_checkpoint(destination_req, buffers, checkpoint_slot)

    assert torch.equal(buffers.kv_buffer[checkpoint_slot], expected_kv)
    assert torch.equal(store.kv_buffer[:, destination_req * window : (destination_req + 1) * window], expected_kv)
    assert store.req_to_seq_len[destination_req].item() == 9


class _TargetStateLayout:
    def __init__(self, page_bytes):
        self.page_bytes = page_bytes

    def get_cpu_cache_big_page_bytes(self):
        return self.page_bytes


def test_cpu_page_keeps_tp_rank_window_states_separate():
    target_page_bytes = 32
    window_config = WindowStateCacheConfig(layers=2, kv_heads=1, head_dim=3, window=4)
    config = WindowedMTPCacheConfig(
        window_config=window_config,
        dtype=torch.float16,
        tp_world_size=2,
        full_att_all_num_kv_heads=0,
        full_att_head_dim=0,
        full_att_layer_num=0,
        linear_config=_TargetStateLayout(target_page_bytes),
    )
    pages = torch.zeros((2, config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8)

    rank0_kv, rank0_ends = config.get_window_views(pages, tp_rank=0)
    rank1_kv, rank1_ends = config.get_window_views(pages, tp_rank=1)
    rank0_kv.fill_(1)
    rank1_kv.fill_(2)
    rank0_ends.copy_(torch.tensor([[11], [12]], dtype=torch.int64))
    rank1_ends.copy_(torch.tensor([[21], [22]], dtype=torch.int64))

    restored_rank0 = config.get_window_views(pages, tp_rank=0)
    restored_rank1 = config.get_window_views(pages, tp_rank=1)
    assert torch.all(restored_rank0[0] == 1)
    assert torch.all(restored_rank1[0] == 2)
    assert restored_rank0[1].tolist() == [[11], [12]]
    assert restored_rank1[1].tolist() == [[21], [22]]
    assert torch.count_nonzero(pages[:, :target_page_bytes]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("window", [1, 4, 8])
def test_window_commit_and_pack_handles_ragged_strided_inputs_and_rejections(window, tensor_cache):
    store = WindowKVStore(3, 2, 1, 3, torch.float32, "cuda", window)
    reqs = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    # Neither the hidden rows nor the noise head dimensions are contiguous.
    hidden = torch.arange(20 * 12, device="cuda", dtype=torch.float32).view(20, 12)[::2, ::2]
    starts = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
    first = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([7, 3], dtype=torch.int32, device="cuda")
    features, positions, mem_index = store.prepare(reqs, hidden, starts, first, lengths, 7, alloc_func=tensor_cache)
    for layer in range(2):
        store.write(layer, mem_index, features.view(-1, 2, 3) + layer)

    accepted_hidden = torch.arange(5 * 6, device="cuda", dtype=torch.float32).view(5, 6) + 1000
    starts = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    first = torch.tensor([7, 3], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    features, positions, mem_index = store.prepare(
        reqs, accepted_hidden, starts, first, lengths, 3, alloc_func=tensor_cache
    )
    for layer in range(2):
        store.write(layer, mem_index, features.view(-1, 2, 3) + layer)

    noise = torch.randn(4, 2, 6, device="cuda")[:, :, ::2]
    history = [torch.cat((hidden[:7], accepted_hidden[:2])), torch.cat((hidden[7:10], accepted_hidden[3:4]))]
    for layer in range(2):
        packed = store.pack(layer, reqs, noise, 2, alloc_func=tensor_cache)
        expected = torch.zeros_like(packed)
        for batch, values in enumerate(history):
            retained = values[-window:].view(-1, 2, 3) + layer
            expected[batch, : len(retained)] = retained
            expected[batch, len(retained) : len(retained) + 2] = noise[batch * 2 : (batch + 1) * 2]
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    assert store.req_to_seq_len.tolist() == [9, 0, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_window_pack_graph_replay_reads_current_request_and_sequence_length(tensor_cache):
    store = WindowKVStore(2, 1, 1, 8, torch.float32, "cuda", 4)
    store.kv_buffer.copy_(torch.arange(8 * 2 * 8, device="cuda").view(1, 8, 2, 8))
    store.req_to_seq_len.copy_(torch.tensor([7, 1], device="cuda"))
    reqs = torch.tensor([0], dtype=torch.int32, device="cuda")
    noise = torch.randn(2, 2, 8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        store.pack(0, reqs, noise, 2, alloc_func=tensor_cache)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = store.pack(0, reqs, noise, 2, alloc_func=tensor_cache)
    reqs.fill_(1)
    graph.replay()
    expected = torch.zeros_like(packed)
    expected[0, 0] = store.kv_buffer[0, 4]
    expected[0, 1:3] = noise
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    store.reset_req(1)
    graph.replay()
    expected.zero_()
    expected[0, :2] = noise
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_window_batch_uses_existing_empty_token_shapes(tensor_cache):
    store = WindowKVStore(1, 1, 1, 8, torch.float32, "cuda", 4)
    reqs = torch.empty(0, dtype=torch.int32, device="cuda")
    features, positions, mem_index = store.prepare(
        reqs, torch.empty(0, 16, device="cuda"), reqs, reqs, reqs, 3, alloc_func=tensor_cache
    )
    assert features.shape == (0, 16)
    assert positions.shape == mem_index.shape == (0,)
    assert store.pack(0, reqs, torch.empty(0, 2, 8, device="cuda"), 2, alloc_func=tensor_cache).shape == (0, 6, 2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tp_world_size", [1, 2, 4])
def test_window_target_pages_reuse_normal_cpu_transfer_with_partial_pages(monkeypatch, tp_world_size):
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
    readies = torch.zeros(1, dtype=torch.int32, device="cuda")
    for rank, kv in enumerate(kvs):
        offload_gpu_kv_to_cpu(mem_index, kv, None, cpu_kv, None, page_index, readies, rank, tp_world_size, 16)
    for rank, kv in enumerate(kvs):
        kv.zero_()
        load_cpu_kv_to_gpu(mem_index, kv, None, cpu_kv, None, page_index, rank, tp_world_size, 16)
        torch.testing.assert_close(kv[:, [1, 5]], expected[rank][:, [1, 5]], rtol=0, atol=0)
        assert not kv[:, [0, 2, 3, 4, 6, 7]].any()
    assert pages[:, config.get_cpu_cache_window_offset() :].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prefill_tp,decode_tp", [(1, 1), (1, 2), (2, 1)])
def test_window_pd_page_reuses_page_io_across_tp_sizes(monkeypatch, prefill_tp, decode_tp):
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

    source, write_helper = make_mems(prefill_tp)
    destination, read_helper = make_mems(decode_tp)
    global_kv = torch.randn(2, 4, 4, 8, device="cuda")

    def shard_heads(rank, tp):
        start = rank * (2 // tp)
        return list(range(start, start + 2 // tp)) + list(range(2 + start, 2 + start + 2 // tp))

    for rank, mem in enumerate(source):
        store = mem.windowed_draft_kv
        store.kv_buffer[:, :4].copy_(global_kv[:, :, shard_heads(rank, prefill_tp)])
        store.req_to_seq_len[0] = 7
    source[0].kv_move_buffer[:, :16].fill_(42)
    write_helper.copy_req_page(0, 0, source, "write")
    destination[0].kv_move_buffer.copy_(source[0].kv_move_buffer)
    for mem in destination:
        mem.windowed_draft_kv.kv_buffer.zero_()
    read_helper.copy_req_page(0, 2, destination, "read")
    for rank, mem in enumerate(destination):
        store = mem.windowed_draft_kv
        torch.testing.assert_close(store.kv_buffer[:, 8:12], global_kv[:, :, shard_heads(rank, decode_tp)])
        assert store.req_to_seq_len.tolist() == [0, 0, 7]
        assert not store.kv_buffer[:, :8].any()
    assert destination[0].kv_move_buffer[:, :16].eq(42).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("linear_target", [False, True])
def test_tail_checkpoint_copy_preserves_stream_order_and_neighbor_slots(linear_target):
    from lightllm.common.kv_cache_mem_manager.operator.windowed_mtp import WindowedMTPMemOperator
    from lightllm.common.kv_cache_mem_manager.operator.linear_att import LinearAttMemOperator

    config = WindowStateCacheConfig(layers=2, kv_heads=1, head_dim=8, window=4)
    source_window = WindowStateBuffers(2, config, torch.bfloat16)
    destination_window = WindowStateBuffers(2, config, torch.bfloat16)
    source_target = tuple(
        torch.zeros((2, 4, 8), dtype=dtype, pin_memory=True) for dtype in (torch.bfloat16, torch.float32)
    )
    destination_target = tuple(torch.zeros_like(state, pin_memory=True) for state in source_target)
    source = SimpleNamespace(
        draft_window=source_window,
        get_state_cache=lambda slot: tuple(state[slot] for state in source_target),
    )
    destination = SimpleNamespace(
        draft_window=destination_window,
        get_state_cache=lambda slot: tuple(state[slot] for state in destination_target),
    )
    operator_class = LinearAttMemOperator if linear_target else WindowedMTPMemOperator
    operator = operator_class.__new__(operator_class)
    operator.mem_manager = SimpleNamespace(big_page_buffers=destination)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the save pending so a host-side tail copy would see stale data.
        torch.cuda._sleep(1_000_000)
        expected_kv = torch.randn(source_window.kv_buffer[1].shape, dtype=torch.bfloat16, device="cuda")
        expected_seq_len = torch.tensor([2 ** 35 + 7], dtype=torch.int64, device="cuda")
        expected_target = [torch.randn(state.shape[1:], dtype=state.dtype, device="cuda") for state in source_target]
        source_window.kv_buffer[1].copy_(expected_kv, non_blocking=True)
        source_window.seq_lens[1:2].copy_(expected_seq_len, non_blocking=True)
        for state, expected in zip(source_target, expected_target):
            state[1].copy_(expected, non_blocking=True)

        operator._copy_tail_state(source, source_id=1, destination_id=0)
        restored_kv = torch.empty_like(expected_kv)
        restored_seq_len = torch.empty_like(expected_seq_len)
        restored_kv.copy_(destination_window.kv_buffer[0], non_blocking=True)
        restored_seq_len.copy_(destination_window.seq_lens[:1], non_blocking=True)
    stream.synchronize()

    torch.testing.assert_close(restored_kv, expected_kv, rtol=0, atol=0)
    torch.testing.assert_close(restored_seq_len, expected_seq_len, rtol=0, atol=0)
    torch.testing.assert_close(destination_window.kv_buffer[0], expected_kv.cpu(), rtol=0, atol=0)
    assert not destination_window.kv_buffer[1].any()
    assert destination_window.seq_lens.tolist() == [2 ** 35 + 7, 0]
    for state, expected in zip(destination_target, expected_target):
        if linear_target:
            torch.testing.assert_close(state[0], expected.cpu(), rtol=0, atol=0)
        else:
            assert not state[0].any()
        assert not state[1].any()
