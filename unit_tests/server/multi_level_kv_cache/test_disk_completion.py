from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("light_mem")

from lightllm.server.multi_level_kv_cache import disk_cache_worker as disk
from lightllm.server.multi_level_kv_cache.manager import MultiLevelKVCacheManager
from lightllm.utils.kv_cache_utils import CpuKVCacheMeta


def worker_for(client, monkeypatch, abort_fails):
    monkeypatch.setattr(disk, "TASK_WAIT_TIMEOUT_S", 0)
    worker = disk.DiskCacheWorker.__new__(disk.DiskCacheWorker)
    worker.cpu_cache_client = client
    worker.max_concurrent_write_tasks = 8
    task = SimpleNamespace(ready=lambda: False, data_safe=lambda: False, page_already_list=[])

    def abort(task):
        if abort_fails:
            raise RuntimeError("I/O worker is unreachable")

    worker.service = SimpleNamespace(
        _n=1,
        query=lambda hashes: [True] * len(hashes),
        create=lambda **kwargs: task,
        abort=abort,
        active_threads=lambda _: 0,
    )
    return worker, task


@pytest.mark.parametrize("abort_fails", [False, True])
def test_disk_read_timeout_only_recycles_after_confirmed_abort(cpu_cache_factory, monkeypatch, abort_fails):
    client = cpu_cache_factory(CpuKVCacheMeta(2, 4096))
    worker, task = worker_for(client, monkeypatch, abort_fails)
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = SimpleNamespace(enable_disk_cache=True)
    manager.cpu_cache_client = client
    manager.disk_cache_worker = worker
    assert manager._disk_cache_match([11, 22], []) == ([], 0)
    allocated, _ = client.allocate_pages([33, 44], False)
    if abort_fails:
        assert allocated == [-1, -1]
        assert worker._retired_tasks == [task]
        assert worker.query_loadable_pages([11, 22], 0) == 0
        assert client.query_one_page(11) == (None, False)

        client.cpu_kv_cache_tensor.fill_(17)
        assert client.any_pages_failed([0, 1])
    else:
        assert all(index >= 0 for index in allocated)


def test_disk_write_abort_failure_keeps_source_lease(cpu_cache_factory, monkeypatch):
    client = cpu_cache_factory(CpuKVCacheMeta(2, 4096))
    pages, _ = client.allocate_pages([11, 22], False)
    client.update_pages_status_to_ready(pages, disk_offload_enable=True, token_num_in_page_list=256)
    worker, task = worker_for(client, monkeypatch, True)
    worker.service.query = lambda hashes: [False] * len(hashes)
    for payload in worker._gather_offload_payloads():
        worker._persist_pages_to_disk(payload)
    assert worker._retired_tasks == [task]
    assert client.any_pages_failed(pages)
    assert client.allocate_pages([33, 44], False)[0] == [-1, -1]


def test_late_completion_does_not_publish_quarantined_page(cpu_cache_factory):
    client = cpu_cache_factory(CpuKVCacheMeta(1, 4096))
    pages, _ = client.allocate_pages([11], False)
    client.update_pages_status_to_ready(pages, deref=False)
    client.query_one_page(11)
    client.quarantine_pages(pages)
    client.update_pages_status_to_ready(pages, deref=True, disk_offload_enable=True, token_num_in_page_list=256)
    assert client.any_pages_failed(pages)
    assert client.query_one_page(11) == (None, False)
    assert client.allocate_pages([22], False)[0] == [-1]
    assert client.get_pages_to_offloading() == []


def test_loading_page_has_one_writer_and_is_not_published_by_a_follower(cpu_cache_factory):
    client = cpu_cache_factory(CpuKVCacheMeta(1, 4096))
    pages, skip_first = client.reserve_write_pages([11], False)
    follower, skip_follower = client.reserve_write_pages([11], False)
    assert pages == follower
    assert skip_first == [False]
    assert skip_follower == [True]
    client.update_pages_status_to_ready(follower, skip_write_flags=skip_follower)
    assert not client.check_allpages_ready(pages)
    assert client.query_one_page(11) == (None, False)
    client.update_pages_status_to_ready(pages, skip_write_flags=skip_first)
    assert client.check_allpages_ready(pages)
    assert client.page_items.linked_items[pages[0]].ref_count == 0


def test_cancelled_writer_invalidates_followers_and_reuses_only_after_last_release(cpu_cache_factory):
    client = cpu_cache_factory(CpuKVCacheMeta(1, 4096))
    pages, _ = client.reserve_write_pages([11], False)
    follower, skip = client.reserve_write_pages([11], False)
    client.recycle_pages(pages)
    assert client.query_one_page(11) == (None, False)
    assert client.any_pages_failed(follower)
    assert client.reserve_write_pages([22], False)[0] == [-1]
    client.update_pages_status_to_ready(follower, skip_write_flags=skip)
    assert client.query_one_page(11) == (None, False)
    assert client.reserve_write_pages([22], False)[0] == pages


@pytest.mark.parametrize("uncertain", [False, True])
def test_disk_block_read_never_writes_published_prefix(cpu_cache_factory, uncertain):
    from lightllm.common.attention_state.lifecycle import UncertainCompletion

    client = cpu_cache_factory(CpuKVCacheMeta(5, 4096))
    prefix, _ = client.reserve_write_pages([11], False)
    client.cpu_kv_cache_tensor[prefix[0]].fill_(9)
    client.update_pages_status_to_ready(prefix, deref=False)
    destinations = []

    def load_pages(hashs, page_indexes, start_pos):
        assert hashs == [11, 22, 33]
        assert start_pos == 0
        destinations.extend(page_indexes)
        assert prefix[0] not in page_indexes
        if uncertain:
            raise UncertainCompletion("read worker is unreachable")
        for index in page_indexes:
            client.cpu_kv_cache_tensor[index].fill_(37)
        return True

    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = SimpleNamespace(enable_disk_cache=True)
    manager.cpu_cache_client = client
    manager.disk_cache_worker = SimpleNamespace(
        service=SimpleNamespace(_n=3), query_loadable_pages=lambda **kwargs: 2, load_pages=load_pages
    )
    loaded, count = manager._disk_cache_match([11, 22, 33], prefix)
    assert loaded[:1] == prefix
    if uncertain:
        assert count == 0
        assert client.any_pages_failed(destinations)

        for index in destinations:
            client.cpu_kv_cache_tensor[index].fill_(19)
    else:
        assert count == 2
        assert client.check_allpages_ready(loaded)
        assert (client.cpu_kv_cache_tensor[loaded[1:]] == 37).all()
    assert (client.cpu_kv_cache_tensor[prefix] == 9).all()


def test_disk_read_does_not_claim_another_writers_loading_page(cpu_cache_factory):
    client = cpu_cache_factory(CpuKVCacheMeta(3, 4096))
    pages, _ = client.reserve_write_pages([11, 22], False)
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = SimpleNamespace(enable_disk_cache=True)
    manager.cpu_cache_client = client
    manager.disk_cache_worker = SimpleNamespace(
        service=SimpleNamespace(_n=1),
        query_loadable_pages=lambda **kwargs: 2,
        load_pages=lambda **kwargs: pytest.fail("the existing writer owns these destinations"),
    )
    assert manager._disk_cache_match([11, 22], []) == ([], 0)
    assert not client.check_allpages_ready(pages)
    assert all(client.page_items.linked_items[index].ref_count == 1 for index in pages)


def test_quarantine_does_not_block_unrelated_free_pages(cpu_cache_factory):
    client = cpu_cache_factory(CpuKVCacheMeta(3, 4096))
    retired, _ = client.reserve_write_pages([11], False)
    client.quarantine_pages(retired)
    client.update_pages_status_to_ready(retired, deref=True)
    for hashes in ([22, 33], [44, 55]):
        pages, _ = client.reserve_write_pages(hashes, False)
        assert all(index >= 0 and index not in retired for index in pages)
        client.update_pages_status_to_ready(pages)


def test_failed_disk_worker_drains_queued_leases_without_freeing_uncertain_io(cpu_cache_factory, monkeypatch):
    client = cpu_cache_factory(CpuKVCacheMeta(2, 4096))
    pages, _ = client.reserve_write_pages([11], False)
    client.update_pages_status_to_ready(pages, disk_offload_enable=True, token_num_in_page_list=256)
    worker, task = worker_for(client, monkeypatch, True)
    first = worker._gather_offload_payloads()[0]
    follower, skip = client.reserve_write_pages([11], False)
    client.update_pages_status_to_ready(
        follower, skip_write_flags=skip, disk_offload_enable=True, token_num_in_page_list=256
    )
    worker.service.query = lambda hashes: [False] * len(hashes)
    worker._persist_pages_to_disk(first)
    assert worker._retired_tasks == [task]
    for payloads in worker._gather_offload_payloads():
        worker._persist_pages_to_disk(payloads)
    assert client.page_items.linked_items[pages[0]].ref_count == 1
    healthy, _ = client.reserve_write_pages([22], False)
    assert healthy[0] >= 0 and healthy != pages


def disk_unavailable(*args, **kwargs):
    raise RuntimeError("disk worker unavailable")


@pytest.mark.parametrize("operation", ["query", "active_threads"])
def test_write_query_failure_releases_unsubmitted_pages(cpu_cache_factory, monkeypatch, operation):
    client = cpu_cache_factory(CpuKVCacheMeta(2, 4096))
    pages, _ = client.reserve_write_pages([11, 22], False)
    client.update_pages_status_to_ready(pages, disk_offload_enable=True, token_num_in_page_list=256)
    worker, _ = worker_for(client, monkeypatch, False)
    worker.service.query = lambda hashes: [False] * len(hashes)
    setattr(worker.service, operation, disk_unavailable)
    worker.service.create = lambda **kwargs: pytest.fail("failed lookup must not submit I/O")
    for payloads in worker._gather_offload_payloads():
        worker._persist_pages_to_disk(payloads)
    assert worker._failed
    assert not client.any_pages_failed(pages)
    assert all(client.page_items.linked_items[index].ref_count == 0 for index in pages)
    assert all(index >= 0 for index in client.reserve_write_pages([33, 44], False)[0])


def test_read_query_failure_keeps_the_valid_cpu_prefix(cpu_cache_factory, monkeypatch):
    client = cpu_cache_factory(CpuKVCacheMeta(2, 4096))
    pages, _ = client.reserve_write_pages([11], False)
    client.update_pages_status_to_ready(pages, deref=False)
    worker, _ = worker_for(client, monkeypatch, False)
    worker.service.query = disk_unavailable
    worker.service.create = lambda **kwargs: pytest.fail("failed lookup must not submit I/O")
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.cpu_cache_client, manager.disk_cache_worker = client, worker
    assert manager._disk_cache_match([11, 22], pages) == (pages, 0)
    assert worker._failed
    assert client.check_allpages_ready(pages)
    assert client.page_items.linked_items[pages[0]].ref_count == 1


def test_completed_read_without_status_is_recomputed_and_recycled(cpu_cache_factory, monkeypatch):
    client = cpu_cache_factory(CpuKVCacheMeta(1, 4096))
    worker, task = worker_for(client, monkeypatch, False)
    task.ready = lambda: True
    task.state = disk_unavailable
    manager = MultiLevelKVCacheManager.__new__(MultiLevelKVCacheManager)
    manager.args = SimpleNamespace(enable_disk_cache=True)
    manager.cpu_cache_client, manager.disk_cache_worker = client, worker
    assert manager._disk_cache_match([11], []) == ([], 0)
    assert worker._failed
    assert client.query_one_page(11) == (None, False)
    assert client.reserve_write_pages([22], False)[0][0] >= 0


def test_write_status_failure_after_submission_retains_every_source(cpu_cache_factory, monkeypatch):
    client = cpu_cache_factory(CpuKVCacheMeta(1, 4096))
    pages, _ = client.reserve_write_pages([11], False)
    client.update_pages_status_to_ready(pages, disk_offload_enable=True, token_num_in_page_list=256)
    worker, _ = worker_for(client, monkeypatch, False)
    worker.service.query = lambda hashes: [False] * len(hashes)

    class UnknownTask:
        page_already_list = property(disk_unavailable)

    task = UnknownTask()
    worker.service.create = lambda **kwargs: task
    for payloads in worker._gather_offload_payloads():
        worker._persist_pages_to_disk(payloads)
    assert worker._retired_tasks == [task]
    assert client.any_pages_failed(pages)
    assert client.reserve_write_pages([22], False)[0] == [-1]
