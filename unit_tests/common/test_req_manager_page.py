from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.common.kv_cache_mem_manager import allocator as allocator_module
from lightllm.common.req_manager import base as req_module
from lightllm.common.req_manager import req_sampling_params
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend import base_backend


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


def _make_scheduler(monkeypatch, page_size, available, batch_max_tokens=8):
    manager = _make_manager(monkeypatch, page_size)
    context = SimpleNamespace(
        req_manager=manager,
        radix_cache=None,
        get_can_alloc_token_num=lambda: available,
        cache_placement_controller=SimpleNamespace(set_req_cache_way=lambda reqs: None),
        filter_reqs=lambda finished_reqs: None,
        pause_reqs=lambda reqs, is_master_in_dp: None,
    )
    monkeypatch.setattr(base_backend, "g_infer_context", context)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.args = SimpleNamespace(enable_cpu_cache=False, enable_prefill_decode_mixed=False, run_mode="normal")
    backend.support_overlap = False
    backend.disable_chunked_prefill = False
    backend.batch_max_tokens = batch_max_tokens
    backend.is_master_in_dp = True
    backend._timer_merge_radix_tree = lambda: None
    backend.prefill_queue_strategy = SimpleNamespace(reorder=lambda reqs, **kwargs: reqs)
    allocations = []
    original_alloc = manager.alloc_req_token_indexes

    def record_alloc(items, radix_cache=None):
        allocations.append(sum(size for _, size in items))
        return original_alloc(items, radix_cache)

    monkeypatch.setattr(manager, "alloc_req_token_indexes", record_alloc)
    return manager, backend, allocations


def _make_scheduler_req(req_idx, current, total):
    req = SimpleNamespace(
        req_idx=req_idx,
        cur_kv_len=current,
        hold_kv_len=0,
        mtp_step=0,
        filter_mark=False,
        wait_pause=False,
        paused=False,
        infer_aborted=False,
        finish_status=SimpleNamespace(is_finished=lambda: False),
        is_slave_req=lambda: False,
        get_cur_total_len=total,
        get_chuncked_input_token_len=total,
    )
    req.prefill_need_token_num = MethodType(InferReq.prefill_need_token_num, req)
    req.decode_need_token_num = MethodType(InferReq.decode_need_token_num, req)
    return req


@pytest.mark.parametrize(
    "hold_kv_len, available, batch_max_tokens, expected_alloc_sizes, second_req_wait_pause",
    [(4, 8, 3, [4], False), (8, 0, 3, [], False), (4, 4, 6, [4], True)],
)
def test_prefill_scheduler_checks_compute_and_kv_budgets_separately(
    monkeypatch, hold_kv_len, available, batch_max_tokens, expected_alloc_sizes, second_req_wait_pause
):
    manager, backend, allocations = _make_scheduler(monkeypatch, 4, available, batch_max_tokens)
    reqs = [_make_scheduler_req(i, 3, lambda: 6) for i in range(2)]
    manager.alloc_req_token_indexes([(req, hold_kv_len) for req in reqs])
    allocations.clear()
    backend._filter_not_ready_reqs = lambda req_ids: reqs

    prefill_reqs, decode_reqs, prefill_tokens = backend._get_classed_reqs(req_ids=[0, 1])

    assert prefill_reqs == [reqs[0]]
    assert decode_reqs == []
    assert prefill_tokens == 3
    assert allocations == expected_alloc_sizes
    assert reqs[0].hold_kv_len == 8
    assert not reqs[0].wait_pause
    assert reqs[1].hold_kv_len == hold_kv_len
    assert reqs[1].wait_pause is second_req_wait_pause
    manager.wait_token_indexes().synchronize()
    assert torch.equal(manager.req_to_token_indexs[0, :8].cpu(), manager.get_cpu_token_indexes(0, 0, 8))


def test_decode_scheduler_publishes_batched_request_table(monkeypatch):
    manager, backend, allocations = _make_scheduler(monkeypatch, 4, 12)
    req = _make_scheduler_req(0, 4, lambda: 5)
    manager.alloc_req_token_indexes([(req, 4)])
    allocations.clear()
    backend._filter_not_ready_reqs = lambda req_ids: [req]

    prefill_reqs, decode_reqs, prefill_tokens = backend._get_classed_reqs(req_ids=[0])

    assert prefill_reqs == []
    assert decode_reqs == [req]
    assert prefill_tokens == 0
    assert req.hold_kv_len == 16
    assert allocations == [12]
    manager.wait_token_indexes().synchronize()
    assert manager.req_to_token_indexs[0, :16].tolist() == list(range(16))


@pytest.mark.parametrize("page_size, expected_alloc_token_num", [(1, 9), (3, 9), (4, 12), (7, 14), (8, 8)])
def test_decode_small_page_preallocation_reuses_one_allocation_for_multiple_steps(
    monkeypatch, page_size, expected_alloc_token_num
):
    manager, backend, allocations = _make_scheduler(monkeypatch, page_size, 32)
    req = _make_scheduler_req(0, page_size, lambda: req.cur_kv_len + 1)
    manager.alloc_req_token_indexes([(req, page_size)])
    allocations.clear()
    backend._filter_not_ready_reqs = lambda req_ids: [req]

    for _ in range(expected_alloc_token_num):
        prefill_reqs, decode_reqs, prefill_tokens = backend._get_classed_reqs(req_ids=[0])
        assert prefill_reqs == []
        assert decode_reqs == [req]
        assert prefill_tokens == 0
        req.cur_kv_len += 1

    assert req.hold_kv_len == page_size + expected_alloc_token_num
    assert allocations == [expected_alloc_token_num]
