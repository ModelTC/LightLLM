from types import SimpleNamespace

import torch

from lightllm.common.kv_cache_mem_manager.windowed_mtp import WindowKVStore
from lightllm.common.state_cache_manager.windowed_mtp import (
    WindowedMTPCacheConfig,
    WindowStateCacheConfig,
)


def _checkpoint_buffers(slots, layers, window, kv_heads, head_dim, dtype):
    return SimpleNamespace(
        kv=torch.zeros((slots, layers, window, 2 * kv_heads, head_dim), dtype=dtype),
        ends=torch.zeros(slots, dtype=torch.int64),
        counts=torch.zeros(slots, dtype=torch.int32),
    )


def test_retained_positions_follow_logical_window_order():
    store = WindowKVStore(
        requests=3,
        layers=1,
        kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
        window=4,
    )
    store.ends.copy_(torch.tensor([0, 3, 7], dtype=torch.int64))
    store.counts.copy_(torch.tensor([0, 3, 4], dtype=torch.int32))

    positions = store.retained_positions(torch.tensor([0, 1, 2], dtype=torch.int64))

    assert torch.equal(
        positions,
        torch.tensor(
            [
                [-1, -1, -1, -1],
                [0, 1, 2, -1],
                [3, 4, 5, 6],
            ],
            dtype=torch.int64,
        ),
    )


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
    store.kv[:, source_req].copy_(expected_kv)
    store.ends[source_req] = 9
    store.counts[source_req] = window

    buffers = _checkpoint_buffers(
        slots=2,
        layers=layers,
        window=window,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=store.kv.dtype,
    )
    store.save_checkpoint(source_req, buffers, checkpoint_slot)

    store.kv[:, destination_req].fill_(-1)
    store.ends[destination_req] = -1
    store.counts[destination_req] = -1
    store.restore_checkpoint(destination_req, buffers, checkpoint_slot)

    assert torch.equal(buffers.kv[checkpoint_slot], expected_kv)
    assert torch.equal(store.kv[:, destination_req], expected_kv)
    assert store.ends[destination_req].item() == 9
    assert store.counts[destination_req].item() == window


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

    rank0_kv, rank0_ends, rank0_counts = config.get_window_views(pages, tp_rank=0)
    rank1_kv, rank1_ends, rank1_counts = config.get_window_views(pages, tp_rank=1)
    rank0_kv.fill_(1)
    rank1_kv.fill_(2)
    rank0_ends.copy_(torch.tensor([[11], [12]], dtype=torch.int64))
    rank1_ends.copy_(torch.tensor([[21], [22]], dtype=torch.int64))
    rank0_counts.copy_(torch.tensor([[3], [4]], dtype=torch.int32))
    rank1_counts.copy_(torch.tensor([[1], [2]], dtype=torch.int32))

    restored_rank0 = config.get_window_views(pages, tp_rank=0)
    restored_rank1 = config.get_window_views(pages, tp_rank=1)
    assert torch.all(restored_rank0[0] == 1)
    assert torch.all(restored_rank1[0] == 2)
    assert restored_rank0[1].tolist() == [[11], [12]]
    assert restored_rank1[1].tolist() == [[21], [22]]
    assert restored_rank0[2].tolist() == [[3], [4]]
    assert restored_rank1[2].tolist() == [[1], [2]]
    assert torch.count_nonzero(pages[:, :target_page_bytes]).item() == 0
