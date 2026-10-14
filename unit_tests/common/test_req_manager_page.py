from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.common.kv_cache_mem_manager import allocator as allocator_module
from lightllm.common.req_manager import base as req_module
from lightllm.common.req_manager import req_sampling_params


class _Counter:
    def __init__(self, *args):
        pass

    def set_value(self, value):
        self.value = value


def _make_manager(monkeypatch, page_size, count=12):
    monkeypatch.setattr(allocator_module, "SharedInt", _Counter)
    monkeypatch.setattr(allocator_module, "get_current_rank_in_node", lambda: 0)
    monkeypatch.setattr(req_module, "get_env_start_args", lambda: SimpleNamespace(page_size=page_size))
    monkeypatch.setattr(req_sampling_params, "ReqSamplingParamsManager", lambda size: None)
    alloc = allocator_module.KvCacheAllocator(page_size * 200, page_size)
    memory = SimpleNamespace(
        page_size=page_size, allocator=alloc, HOLD_TOKEN_MEMINDEXES=tuple(range(alloc.size, alloc.size + page_size))
    )
    return req_module.ReqManager(count, page_size * 10, memory)


@pytest.mark.parametrize("page_size", [1, 2, 4, 8, 64])
@pytest.mark.parametrize("count", [0, 1, 4, 5, 9])
def test_alloc_publish_reuse_and_owned_snapshot(monkeypatch, page_size, count):
    manager = _make_manager(monkeypatch, page_size)
    requests = [SimpleNamespace(req_idx=i, hold_kv_len=0, cur_kv_len=0) for i in range(count)]
    allocations = [(req, page_size * (1 + i % 3)) for i, req in enumerate(requests)]
    manager.alloc_req_token_indexes(allocations)
    torch.cuda.synchronize()
    expected_offset = 0
    snapshots = []
    for req, size in allocations:
        expected = torch.arange(expected_offset, expected_offset + size, dtype=torch.int32)
        assert torch.equal(manager.req_to_token_indexs[req.req_idx, :size].cpu(), expected)
        snapshot = manager.get_cpu_token_indexes(req.req_idx, 0, size)
        assert torch.equal(snapshot, expected)
        snapshots.append((snapshot, expected))
        assert req.hold_kv_len == size
        expected_offset += size
    blocks = [manager.get_cpu_token_index_blocks(req.req_idx, 0, size).copy() for req, size in allocations]
    if blocks:
        manager.free_token_index_blocks(np.concatenate(blocks))
        for req, _ in allocations:
            req.hold_kv_len = 0
        manager.alloc_req_token_indexes(list(reversed(allocations)))
        torch.cuda.synchronize()
        for snapshot, expected in snapshots:
            assert torch.equal(snapshot, expected)
    for req, size in allocations:
        start = min(1, size)
        assert torch.equal(
            manager.get_cpu_token_indexes(req.req_idx, start, size),
            manager.req_to_token_indexs[req.req_idx, start:size].cpu(),
        )


@pytest.mark.parametrize("page_size", [1, 4, 64])
def test_prefix_cross_stream_publish(monkeypatch, page_size):
    manager = _make_manager(monkeypatch, page_size)
    writer, reader = torch.cuda.Stream(), torch.cuda.Stream()
    source = torch.arange(page_size * 3, dtype=torch.int32) + page_size * 20
    with torch.cuda.stream(writer):
        prefixes = {}
        manager.write_token_index_prefix(0, 0, source, prefixes)
        manager.copy_token_index_prefixes_to_gpu(prefixes)
    with torch.cuda.stream(reader):
        manager.copy_token_index_prefix(0, 1, len(source))
        observed = manager.req_to_token_indexs[1, : len(source)].clone()
    reader.synchronize()
    assert torch.equal(observed.cpu(), source)
    assert torch.equal(manager.get_cpu_token_indexes(1, 0, len(source)), source)


@pytest.mark.parametrize("page_size", [1, 4, 64])
def test_alloc_token_indexes_returns_logical_span_and_reuses_page_tail(monkeypatch, page_size):
    manager = _make_manager(monkeypatch, page_size)
    req = SimpleNamespace(req_idx=0, hold_kv_len=0, cur_kv_len=0)
    indexes = manager.alloc_token_indexes(req, page_size + 1)
    assert indexes.tolist() == list(range(page_size + 1))
    expected_hold = (page_size + 1 + page_size - 1) // page_size * page_size
    assert req.hold_kv_len == expected_hold
    capacity = manager.mem_manager.allocator.can_use_mem_size
    req.cur_kv_len = page_size
    tail = manager.alloc_token_indexes(req, expected_hold)
    assert tail.tolist() == list(range(page_size, expected_hold))
    assert manager.mem_manager.allocator.can_use_mem_size == capacity
