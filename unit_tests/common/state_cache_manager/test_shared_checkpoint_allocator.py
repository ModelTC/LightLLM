import multiprocessing as mp
import os
import uuid
from types import SimpleNamespace as NS

import pytest
import numpy as np
import torch
from sortedcontainers import SortedDict, SortedSet

from lightllm.common.state_cache_manager.base import StateCacheManager
from lightllm.utils.envs_utils import get_unique_server_name


class SlotPool(StateCacheManager):
    def get_state_cache(self, buffer_idx):
        raise NotImplementedError()


def _service_name(name):
    os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = name
    get_unique_server_name.cache_clear()


def _attach(name, create=False):
    return SlotPool(10, keep_num=2, shm_name=name, shm_create=create)


@pytest.fixture
def pool(monkeypatch):
    name = f"test_checkpoints_{uuid.uuid4().hex}"
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", name)
    get_unique_server_name.cache_clear()
    manager = _attach("pool", create=True)
    yield manager
    manager.metadata_shm.unlink()
    manager.lock.shm.unlink()
    get_unique_server_name.cache_clear()


def test_private_pool_allocation_and_reserved_slots():
    manager = SlotPool(4, keep_num=2)
    assert manager.alloc_state_cache(2) == [0, 1]
    assert manager.alloc_one_state_cache() is None
    manager.free_state_cache([0])
    assert manager.alloc_one_state_cache() == 0
    with pytest.raises(AssertionError, match="out of alloc range"):
        manager.free_state_cache([2])


def test_private_pool_reuses_released_slots_in_fifo_order():
    manager = SlotPool(5, keep_num=1)
    assert manager.alloc_state_cache(2) == [0, 1]
    manager.free_state_cache([0])
    reused = manager.alloc_state_cache(3)
    assert reused == [2, 3, 0]
    manager.free_state_cache([1] + reused)
    assert manager.get_used_cache_num() == 1
    assert (manager.refs == 0).all()
    assert manager.alloc_state_cache(4) == [1, 2, 3, 0]
    assert manager.alloc_one_state_cache() is None


@pytest.mark.parametrize("shared", [False, True])
def test_fifo_wraparound_preserves_release_order(pool, shared):
    source = pool if shared else SlotPool(10, keep_num=2)
    target = _attach("pool") if shared else source
    assert source.alloc_state_cache(3) == [0, 1, 2]
    source.free_state_cache([1, 0])
    assert target.alloc_state_cache(8) is None
    assert target.alloc_state_cache(0) == []
    assert target.alloc_state_cache(5) == [3, 4, 5, 6, 7]
    assert target.alloc_state_cache(2) == [1, 0]
    assert source.get_free_cache_num() == 0
    assert source.alloc_one_state_cache() is None

    release_order = [2, 7, 6, 5, 4, 3, 0, 1]
    source.free_state_cache(release_order)
    assert target.alloc_state_cache(8) == release_order
    assert source.get_free_cache_num() == 0
    assert (source.refs == 1).all()


def test_source_release_does_not_reclaim_destination_checkpoint(pool):
    target = _attach("pool")
    slot = pool.alloc_one_state_cache()
    target.retain_state_cache([slot])
    assert pool.get_free_cache_num() == 7
    pool.free_state_cache([slot])
    assert target.refs[slot] == 1
    other_slots = pool.alloc_state_cache(7)
    assert slot not in other_slots
    assert pool.alloc_one_state_cache() is None
    target.free_state_cache([slot])
    assert pool.alloc_one_state_cache() == slot
    pool.free_state_cache(other_slots + [slot])
    assert pool.get_used_cache_num() == 2


def test_startup_attach_preserves_metadata(pool):
    target = _attach("pool")
    slot = pool.alloc_one_state_cache()
    with pool.lock:
        assert target.lock.shm.buf.cast("i")[0] == 1
        assert target.refs[slot] == 1
    assert target.get_free_cache_num() == 7
    with pytest.raises(AssertionError, match="out of alloc range"):
        target.free_state_cache([8])
    pool.free_state_cache([slot])
    with pytest.raises(AssertionError, match="has no owner"):
        target.free_state_cache([slot])


def _allocator_worker(service_name, attached, ready, release):
    _service_name(service_name)
    manager = _attach("pool")
    attached.wait(timeout=60)
    slots = manager.alloc_state_cache(4)
    ready.put(slots)
    assert release.wait(30)
    manager.free_state_cache(slots)
    for _ in range(300):
        slot = manager.alloc_one_state_cache()
        assert slot is not None
        manager.retain_state_cache([slot])
        manager.free_state_cache([slot])
        with manager.lock:
            assert manager.refs[slot] == 1
        manager.free_state_cache([slot])


def test_allocator_and_refs_are_process_shared(pool):
    context = mp.get_context("spawn")
    ready, release = context.Queue(), context.Event()
    attached = context.Barrier(2)
    workers = [
        context.Process(target=_allocator_worker, args=(get_unique_server_name(), attached, ready, release))
        for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        first, second = ready.get(timeout=60), ready.get(timeout=60)
        assert len(set(first + second)) == 8
        assert pool.get_free_cache_num() == 0
        release.set()
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0
        assert pool.get_free_cache_num() == 8
        assert (pool.refs == 0).all()
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)


def _radix_cache(pool):
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import (
        HybridAttPagedRadixCache,
        HybridAttPagedTreeNode,
    )

    cache = object.__new__(HybridAttPagedRadixCache)
    cache.hash_page_size, cache.big_page_num, cache.big_page_tokens = 2, 2, 4
    cache.big_page_buffers = pool
    cache.small_page_buffers = SlotPool(2)
    cache.refed_tokens_num = NS(arr=np.zeros(1, dtype=np.int64))
    cache.tree_total_tokens_num = NS(arr=np.zeros(1, dtype=np.int64))
    cache._evict_tree_set = SortedSet(key=lambda node: node.get_compare_key())
    cache._evict_tree_set_for_state_cache = SortedSet(key=lambda node: node.get_compare_key_for_buffer_idx())
    root = cache.root_node = HybridAttPagedTreeNode(2, 2)
    root.ref_counter, root.page_num = 1, 2
    root.token_mem_index_value = torch.empty(0, dtype=torch.int64)
    return cache


def test_radix_ownership_transfer_duplicates_and_source_eviction(pool):
    source, target = _radix_cache(pool), _radix_cache(_attach("pool"))
    slot = pool.alloc_one_state_cache()
    key = torch.arange(4)
    source.insert(key, key, [11, 17], [None, None], SortedDict({4: slot}))
    source_node, _, _ = source.match_prefix(key, [11, 17], update_refs=True)
    target.big_page_buffers.retain_state_cache([slot])
    incoming = SortedDict({4: slot})
    _, target_node = target.insert(key, key, [11, 17], [None, None], incoming)
    assert not incoming and target_node.big_page_buffer_idx == slot
    assert pool.refs[slot] == 2
    source.dec_node_ref_counter(source_node)
    source._evict(4, lambda *args: None)
    assert pool.refs[slot] == 1
    assert not source.root_node.children and target.root_node.children

    # The same ID can arrive again. Only the incoming request ref is released on deduplication.
    pool.retain_state_cache([slot])
    duplicate = SortedDict({4: slot})
    target.insert(key, key, [11, 17], [None, None], duplicate)
    assert not duplicate and pool.refs[slot] == 1
    target._evict(4, lambda *args: None)
    assert pool.get_free_cache_num() == 8


def test_aborted_request_releases_only_its_adopted_reference(pool):
    from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    slot = pool.alloc_one_state_cache()
    pool.retain_state_cache([slot])
    context = InferenceContext()
    context.is_hybrid_att_model = True
    context.req_manager = NS(req_to_token_indexs=torch.tensor([[0, 1, 2, 3]]))
    context.radix_cache = _radix_cache(_attach("pool"))
    req = NS(
        req_idx=0,
        hold_kv_len=4,
        shared_kv_node=None,
        tail_small_page_buffer_id=None,
        hybrid_len_to_big_page_id=SortedDict({4: slot}),
    )
    freed_tokens = []
    context._free_req_mem_without_radix_insert(freed_tokens, req)
    assert not req.hybrid_len_to_big_page_id
    assert freed_tokens[0].tolist() == [0, 1, 2, 3]
    assert pool.refs[slot] == 1
    pool.free_state_cache([slot])
    assert pool.get_free_cache_num() == 8
