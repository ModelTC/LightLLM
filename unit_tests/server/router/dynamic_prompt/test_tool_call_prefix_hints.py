import hashlib
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.dynamic_prompt import tool_call_prefix_hints as hints_module
from lightllm.server.router.dynamic_prompt.tool_call_prefix_hints import (
    MAX_PREFIX_HINTS,
    PREFIX_HINT_SHAPE,
    PrefixHintSnapshot,
    ToolCallPrefixHints,
    fingerprint,
    prefix_hasher,
)
from lightllm.utils import shm_utils


@pytest.fixture(autouse=True)
def service_name(monkeypatch):
    monkeypatch.setattr(hints_module, "MIN_PREFIX_HINT_TOKENS", 1)
    name = f"prefix_hints_{uuid.uuid4().hex}"
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: name)


def tree():
    result = RadixCache(1000, 0)
    result.enable_tool_call_retention(10, 0.5)
    result.tool_call_retention.advance_clock(100)
    return result


def protect(cache, tokens):
    node = cache.insert(torch.tensor(tokens))[1]
    cache.tool_call_retention.protect(node)
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    return node


def test_fingerprint_reuse_requires_matching_tokens_before_a_new_tail():
    cache = tree()
    protect(cache, [1, 2, 3])
    snapshot = cache.tool_call_prefix_hints.snapshot(now=101)
    assert snapshot.matches([1, 2, 3, 4])
    assert not snapshot.matches([1, 9, 3, 4])
    assert not snapshot.matches([1, 2, 3])
    assert not snapshot.matches([1, 2])
    assert cache.get_refed_tokens_num() == 0


def test_cached_prefix_fingerprint_survives_radix_split_and_merge():
    cache = tree()
    node = protect(cache, [1, 2, 3, 4])
    expected = fingerprint(prefix_hasher(node))
    cache.match_prefix(torch.tensor([1, 2, 9]))
    assert fingerprint(prefix_hasher(node)) == expected
    cache.tool_call_retention.advance_clock(111)
    cache.merge_unreferenced_nodes()
    assert fingerprint(prefix_hasher(node)) == expected
    raw = np.asarray([1, 2, 3, 4], dtype="<i8").tobytes()
    assert expected == fingerprint(hashlib.sha256(raw))


def test_expiry_ignores_stale_published_entries_without_backend_refresh():
    cache = tree()
    protect(cache, [1, 2, 3])
    assert not cache.tool_call_prefix_hints.snapshot(now=111).matches([1, 2, 3, 4])


def test_consumed_or_evicted_lease_is_removed_on_publish():
    cache = tree()
    protect(cache, [1, 2, 3])
    node, _, _ = cache.match_prefix(torch.tensor([1, 2, 3, 4]), update_refs=True)
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    assert not cache.tool_call_prefix_hints.snapshot(now=101).by_length
    cache.dec_node_ref_counter(node)
    protect(cache, [7, 8, 9])
    cache.evict(6, lambda _: None)
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    assert not cache.tool_call_prefix_hints.snapshot(now=101).by_length


def test_active_endpoint_is_not_published_as_waiting():
    cache = tree()
    node = cache.insert(torch.tensor([1, 2, 3]))[1]
    cache.add_node_ref_counter(node)
    cache.tool_call_retention.protect(node)
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    assert not cache.tool_call_prefix_hints.snapshot(now=101).by_length


def test_odd_writer_generation_fails_closed():
    cache = tree()
    protect(cache, [1, 2, 3])
    cache.tool_call_prefix_hints.shared.arr[0, 0] |= 1
    assert not cache.tool_call_prefix_hints.snapshot(now=101).by_length


def test_table_is_bounded_with_many_distinct_waiters():
    cache = tree()
    for i in range(MAX_PREFIX_HINTS + 10):
        protect(cache, [i + 1])
    assert cache.tool_call_prefix_hints.shared.arr[0, 1] == MAX_PREFIX_HINTS
    assert cache.tool_call_prefix_hints.shared.arr.shape == PREFIX_HINT_SHAPE


def test_request_memo_reuses_fingerprints_when_lease_deadline_changes():
    state = hashlib.sha256(np.asarray([1, 2, 3], dtype="<i8").tobytes())
    first, second = fingerprint(state)
    memo = {}
    assert PrefixHintSnapshot([(3, first, second, 110000000)], 100).matches([1, 2, 3, 4], memo)
    assert memo.keys() == {3}
    assert PrefixHintSnapshot([(3, first, second, 120000000)], 115).matches([1, 2, 3, 4], memo)
    assert not PrefixHintSnapshot([(3, first, second, 110000000)], 115).matches([1, 2, 3, 4], memo)


def test_reader_attaches_existing_publisher_without_resetting():
    cache = tree()
    protect(cache, [1, 2, 3])
    reader = ToolCallPrefixHints(0)
    assert reader.snapshot(now=101).matches([1, 2, 3, 4])


def test_tiny_shared_scaffolding_does_not_prioritize_unrelated_requests(monkeypatch):
    monkeypatch.setattr(hints_module, "MIN_PREFIX_HINT_TOKENS", 256)
    cache = tree()
    protect(cache, list(range(255)))
    assert not cache.tool_call_prefix_hints.snapshot(now=101).by_length
    protect(cache, list(range(256)))
    assert cache.tool_call_prefix_hints.snapshot(now=101).matches(list(range(257)))


def test_new_hint_lengths_cannot_grow_request_memo_past_bound():
    ids = np.arange(131, dtype=np.int64)
    first, second = fingerprint(hashlib.sha256(ids[:129].astype("<i8").tobytes()))
    memo = {length: (0, 0) for length in range(1, 129)}
    assert PrefixHintSnapshot([(129, first, second, 110000000)], 100).matches(ids, memo)
    assert len(memo) <= MAX_PREFIX_HINTS * 2


def test_hybrid_admission_hint_ends_at_a_real_budgeted_checkpoint():
    from sortedcontainers import SortedDict
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache

    buffers = SimpleNamespace(free_state_cache=lambda *args, **kwargs: None)
    cache = HybridAttPagedRadixCache(20, 0, 2, 2, SimpleNamespace(big_page_buffers=buffers), buffers)
    cache.enable_tool_call_retention(10, 0.25)
    cache.tool_call_retention.advance_clock(100)
    _, tail = cache.insert(
        torch.arange(12),
        block_hashs=[11, 22, 33, 44, 55, 66],
        block_state_idxs=[None] * 6,
        len_to_big_page_id=SortedDict({4: 100, 8: 101, 12: 102}),
    )
    cache.tool_call_retention.protect(tail)
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    snapshot = cache.tool_call_prefix_hints.snapshot(now=101)
    assert set(snapshot.by_length) == {4}
    assert snapshot.matches(np.arange(13))
    cache.match_prefix(torch.arange(12), block_hashs=[11, 22, 33, 44, 55, 66], update_refs=True)
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    assert not cache.tool_call_prefix_hints.snapshot(now=101).by_length


def test_lost_hybrid_state_removes_admission_hint_when_published():
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache

    freed = []
    buffers = SimpleNamespace(
        free_state_cache=lambda indexes=None, free_indexes=None: freed.extend(indexes or free_indexes),
        get_free_cache_num=lambda: 0,
    )
    cache = HybridAttPagedRadixCache(40, 0, 2, 2, SimpleNamespace(big_page_buffers=buffers), buffers)
    cache.enable_tool_call_retention(10, 0.5)
    cache.tool_call_retention.advance_clock(100)
    _, node = cache.insert(torch.tensor([1, 2]), block_hashs=[11], block_state_idxs=[10])
    cache.tool_call_retention.protect(node)
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    assert cache.tool_call_prefix_hints.snapshot(now=101).matches([1, 2, 3])
    cache.free_one_small_page_buffer()
    cache.tool_call_prefix_hints.publish(cache.tool_call_retention)
    assert freed == [10]
    assert not cache.tool_call_prefix_hints.snapshot(now=101).by_length
