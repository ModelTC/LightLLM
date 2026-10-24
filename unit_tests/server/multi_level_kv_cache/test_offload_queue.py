from types import SimpleNamespace

import numpy as np
import torch

from lightllm.server.multi_level_kv_cache import shm_objs
from lightllm.utils.kv_cache_utils import CpuKVCacheMeta


def test_int_list_rejects_overflow_without_changing_contents(monkeypatch):
    monkeypatch.setattr(
        shm_objs, "_create_shm", lambda name, byte_size: SimpleNamespace(size=byte_size, buf=bytearray(byte_size))
    )
    queue = shm_objs.IntList("offload", capacity=4, init_shm_data=True)
    queue.add_items([1, 2, 3])
    expected = queue.arr.copy()
    assert not queue.add_items([4, 5])
    np.testing.assert_array_equal(queue.arr, expected)
    queue.add_item(4)
    expected = queue.arr.copy()
    assert not queue.add_item(5)
    np.testing.assert_array_equal(queue.arr, expected)
    assert queue.pop_all_item() == [1, 2, 3, 4]
    assert queue.pop_all_item() is None
    queue.add_items([6, 7])
    assert queue.pop_all_item() == [6, 7]


def test_repeated_disk_offloads_preserve_refs_when_queue_is_full(cpu_cache_factory):
    client = cpu_cache_factory(CpuKVCacheMeta(4, 1, 1, 1, 8, torch.uint8, 0, torch.uint8))
    hashes = [11, 22, 33, 44]
    for _ in range(3):
        pages, _ = client.allocate_pages(hashes, disk_offload_enable=True)
        client.update_pages_status_to_ready(pages, disk_offload_enable=True, token_num_in_page_list=512)
        assert client.offload_page_indexes.size() == 5
        assert [client.page_items.linked_items[page].ref_count for page in pages] == [1] * 4
        assert all(client.page_items.linked_items[page].is_ready() for page in pages)

    assert client.get_pages_to_offloading() == [pages]
    assert client.get_pages_to_offloading() == []
    client.deref_pages(pages)
    assert [client.page_items.linked_items[page].ref_count for page in pages] == [0] * 4
    item = client.page_items.head
    for page in pages:
        item = item.get_next_item()
        assert item.self_index == page
    assert item.get_next_item().self_index == client.page_items.tail.self_index

    reused, ready = client.allocate_pages(hashes, disk_offload_enable=True)
    assert reused == pages and all(ready)
    client.update_pages_status_to_ready(reused, disk_offload_enable=True, token_num_in_page_list=512)
    assert client.get_pages_to_offloading() == [pages]
    client.deref_pages(pages)
    replaced, ready = client.allocate_pages([55, 66, 77, 88], disk_offload_enable=True)
    assert replaced == pages and not any(ready)
    assert all(client.page_hash_dict.get(key) is None for key in hashes)
