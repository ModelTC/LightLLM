from collections import deque
from types import SimpleNamespace as NS

import pytest
import torch

from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as cache_module
from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import (
    Dsv4CpuCacheSession,
    Dsv4LoadTask,
    Dsv4StagingSlot,
    Dsv4StorePage,
    Dsv4StoreTask,
    MultiLevelKvCacheModule,
)
from unit_tests.common.test_deepseek4_paged_cache import cache as _cache


cache = _cache


class Event:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done

    def record(self):
        pass

    def synchronize(self):
        self.done = True


def values(items):
    return NS(get_all=lambda: list(items))


class Client:
    def __init__(self):
        self.lock = NS(acquire_sleep1ms=lambda: None, release=lambda: None)
        self.released, self.published = [], []
        self.ready = set()

    def deref_pages(self, page_list):
        self.released.extend(page_list)

    def check_allpages_ready(self, pages):
        return all(page in self.ready for page in pages)

    def update_pages_status_to_ready(self, page_list, **kwargs):
        self.ready.update(page_list)
        self.published.append((list(page_list), kwargs))
        if kwargs.get("deref"):
            self.deref_pages(page_list)


def module():
    result = object.__new__(MultiLevelKvCacheModule)
    result.args = NS(cpu_cache_token_page_size=2048, enable_disk_cache=False, page_size=256)
    result.backend = NS(is_deepseek_v4=True, is_master_in_dp=True)
    result.cpu_cache_client = Client()
    result.dsv4_sessions = {}
    result.dsv4_load_tasks = deque()
    result.dsv4_store_tasks = deque()
    result.dsv4_staging_slots = [Dsv4StagingSlot(), Dsv4StagingSlot()]
    result.dsv4_max_pages_per_task = 4
    return result


def test_load_leases_survive_request_release_until_cuda_event_completes():
    obj = module()
    event = Event()
    session = Dsv4CpuCacheSession(7, leased_pages=[11, 12], load_event=event)
    obj.dsv4_sessions[7] = session
    req = NS(req_id=7)
    assert obj.offload_finished_reqs_to_cpu_cache([req]) == [req]
    assert obj.cpu_cache_client.released == []
    event.done = True
    obj.update_cpu_cache_task_states()
    assert obj.cpu_cache_client.released == [12, 11]
    assert obj.dsv4_sessions == {}


def test_finished_request_does_not_wait_for_store_or_survive_inside_task():
    obj = module()
    session = Dsv4CpuCacheSession(7, leased_pages=[11], pending_tasks=1)
    obj.dsv4_sessions[7] = session
    event, released = Event(), []
    pool = NS(free_state_cache=lambda ids: released.extend(ids))
    slot = Dsv4StagingSlot(in_use=True)
    obj.dsv4_store_tasks.append(Dsv4StoreTask([11], [session], slot, event, [(pool, [3])]))
    req = NS(req_id=7)
    assert obj.offload_finished_reqs_to_cpu_cache([req]) == [req]
    # Teardown/reuse of the request object cannot change the offload metadata.
    req.req_id = 99
    assert not obj.cpu_cache_client.published and not released
    event.done = True
    obj.update_cpu_cache_task_states()
    obj.update_cpu_cache_task_states()
    assert obj.cpu_cache_client.published == [([11], {"deref": False})]
    assert obj.cpu_cache_client.released == [11]
    assert released == [3] and not slot.in_use
    assert obj.dsv4_sessions == {}


def test_nonmaster_abort_cannot_recycle_a_loading_checkpoint():
    obj = module()
    obj.backend.is_master_in_dp = False
    event, released = Event(), []
    pool = NS(free_state_cache=lambda ids: released.extend(ids))
    obj.dsv4_load_tasks.append(Dsv4LoadTask(event, [(pool, [4])]))
    req = NS(req_id=7)
    assert obj.offload_finished_reqs_to_cpu_cache([req]) == [req]
    obj.update_cpu_cache_task_states()
    assert released == []
    event.done = True
    obj.update_cpu_cache_task_states()
    obj.update_cpu_cache_task_states()
    assert released == [4] and obj.cpu_cache_client.released == []


def test_disk_publication_waits_for_other_owner_and_preserves_prefix_order():
    obj = module()
    session = Dsv4CpuCacheSession(7, leased_pages=[11, 3, 8], token_num=4864, closing=True, disk_offload_enable=True)
    obj.dsv4_sessions[7] = session
    obj.cpu_cache_client.ready.update([11, 8])
    obj.update_cpu_cache_task_states()
    assert not obj.cpu_cache_client.published and not obj.cpu_cache_client.released
    obj.cpu_cache_client.ready.add(3)
    obj.update_cpu_cache_task_states()
    assert obj.cpu_cache_client.published == [
        ([11, 3, 8], {"deref": True, "disk_offload_enable": True, "token_num_in_page_list": 4864})
    ]


def test_store_allocations_do_not_overwrite_another_loading_owner():
    obj = object.__new__(CpuKvCacheClient)
    obj.page_items = NS(linked_items=[])
    obj.page_hash_dict = {1: 10, 2: 20}

    def allocate(page_items, hash_key, disk_offload_enable):
        page = obj.page_hash_dict.setdefault(hash_key, hash_key * 10)
        return page, hash_key == 2

    obj.allocate_one_page = allocate
    assert obj.allocate_pages([1, 2, 3, 3], False, return_ownership=True) == (
        [10, 20, 30, 30],
        [False, False, True, False],
    )
    assert obj.allocate_pages([1, 2, 3], False) == ([10, 20, 30], [False, True, False])


@pytest.mark.parametrize("is_dsv4", [False, True])
def test_loader_uses_events_for_dsv4_and_keeps_other_models_synchronous(monkeypatch, is_dsv4):
    obj = module()
    obj.backend.is_deepseek_v4 = is_dsv4
    obj.init_sync_group = None
    obj.need_sync_compute_stream = lambda: False
    calls = []
    req = NS(
        req_id=7,
        req_idx=0,
        cur_kv_len=0,
        hold_kv_len=0,
        mm_input_spans=[],
        sampling_param=NS(shm_param=NS(prompt_logprobs=-1)),
        shm_req=NS(
            cpu_cache_match_page_indexes=values([11]),
            token_hash_page_len_list=values([2048]),
            disk_prompt_cache_len=0,
            cpu_prompt_cache_len=0,
            input_len=4096,
        ),
    )
    obj.backend.model = NS(
        req_manager=NS(req_to_token_indexs=torch.arange(4096).view(1, -1)),
        mem_manager=NS(operator=NS(load_cpu_cache_to_gpu=lambda **kwargs: calls.append("load") or [])),
    )
    obj.backend._alloc_req_kv_mem = lambda req, count: torch.arange(count, dtype=torch.int32)
    monkeypatch.setattr(cache_module.g_infer_context, "get_can_alloc_token_num", lambda: 4096)
    monkeypatch.setattr(cache_module.dist, "barrier", lambda **kwargs: calls.append("barrier"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: NS(synchronize=lambda: calls.append("sync")))
    obj.load_cpu_cache_to_reqs([req])
    assert req.cur_kv_len == 2048
    assert calls == (["load", "barrier"] if is_dsv4 else ["load", "sync", "barrier"])
    assert obj.cpu_cache_client.released == ([] if is_dsv4 else [11])
    if is_dsv4:
        assert obj.dsv4_sessions[7].load_event is not None


@pytest.mark.parametrize("owners", [[False, True, False], [True, True, True]])
def test_incremental_store_only_packs_new_owners_and_pins_checkpoints(monkeypatch, owners):
    obj = module()
    retained, submitted = [], []
    pool = NS(retain_state_cache=lambda ids: retained.extend(ids))
    obj.backend.model = NS(
        mem_manager=NS(big_page_buffers=pool), req_manager=NS(req_to_token_indexs=torch.arange(8192).view(1, -1))
    )
    obj.cpu_cache_client.allocate_pages = lambda hashes, **kwargs: ([11, 12, 13], owners)
    obj.dsv4_max_pages_per_task = 1
    obj._handle_hybrid_att_last_page = lambda req, target, lens: target
    monkeypatch.setattr(cache_module.g_infer_context, "radix_cache", NS(get_big_page_ids_by_node=lambda _: []))
    session = Dsv4CpuCacheSession(7)
    obj.dsv4_sessions[7] = session
    req = NS(
        req_id=7,
        req_idx=0,
        cur_kv_len=6144,
        shared_kv_node=None,
        hybrid_len_to_big_page_id={2048: 2, 4096: 4, 6144: 6},
        shm_req=NS(
            group_req_id=7,
            request_id=7,
            token_hash_list=values([1, 2, 3]),
            token_hash_page_len_list=values([2048, 4096, 6144]),
        ),
    )

    def submit(pages, producer_stream):
        # Do not release the reserved owner before its task has been queued.
        assert not session.closing
        obj._poll_dsv4_store_tasks()
        assert obj.cpu_cache_client.released == []
        submitted.extend(pages)
        session.pending_tasks += 1

    obj._submit_dsv4_store_batch = submit
    obj.store_completed_prefill_pages([req], None)
    assert [page.cpu_page_index for page in submitted] == [11 + i for i, owner in enumerate(owners) if owner]
    for page in submitted:
        offset = (page.cpu_page_index - 11) * 2048
        assert page.source_mem_indexes.tolist() == list(range(offset, offset + 2048))
    assert retained == [2 * (i + 1) for i, owner in enumerate(owners) if owner]
    assert session.pending_tasks == sum(owners) and session.leased_pages == [11, 12, 13]
    assert session.token_num == 6144 and session.closing


def test_staging_separates_gpu_request_lifetime_and_handles_two_small_pages(cache, monkeypatch):
    manager, requests = cache
    obj = module()
    client = obj.cpu_cache_client
    obj.backend.model = NS(mem_manager=manager, req_manager=requests)
    requests.create_small_page_cache_manager(4)
    page_tokens = obj.args.cpu_cache_token_page_size
    page_bytes = manager.cache_config.get_cpu_cache_big_page_bytes()
    client.cpu_kv_cache_tensor = torch.zeros((5, page_bytes), dtype=torch.uint8, pin_memory=True)
    cpu_stream, producer = torch.cuda.Stream(), torch.cuda.Stream()
    monkeypatch.setattr(cache_module.g_infer_context, "get_cpu_kv_cache_stream", lambda: cpu_stream)
    for buffer in manager.kv_buffer:
        buffer.random_(0, 256)
    pages, expected = [], []
    for req_id, (length, cpu_id, value) in enumerate([(2048, 2, 11), (768, 4, 22), (1280, 1, 33)]):
        pool = manager.big_page_buffers if length == page_tokens else requests.small_page_buffers
        state_id = pool.alloc_one_state_cache()
        pool.buffer[state_id].fill_(value)
        pool.retain_state_cache([state_id])
        session = Dsv4CpuCacheSession(req_id, leased_pages=[cpu_id])
        obj.dsv4_sessions[req_id] = session
        indexes = torch.arange(req_id * page_tokens, req_id * page_tokens + length, dtype=torch.int32, device="cuda")
        pages.append(Dsv4StorePage(session, cpu_id, indexes, pool, state_id))
        expected.append([buffer[indexes.long()].flatten().cpu() for buffer in manager.kv_buffer])
    producer.wait_stream(torch.cuda.current_stream())
    # Compile/allocate once so initialization does not consume the delayed-copy window.
    obj._submit_dsv4_store_batch(pages, producer)
    cpu_stream.synchronize()
    obj.update_cpu_cache_task_states()
    client.published.clear()
    client.ready.clear()
    for page in pages:
        page.checkpoint_pool.retain_state_cache([page.checkpoint_id])
    with torch.cuda.stream(cpu_stream):
        torch.cuda._sleep(500_000_000)
    obj._submit_dsv4_store_batch(pages, producer)
    assert not obj.dsv4_store_tasks[0].store_event.query()
    reqs = [NS(req_id=page.session.request_id) for page in pages]
    assert obj.offload_finished_reqs_to_cpu_cache(reqs) == reqs
    assert client.published == []
    # Return the original request-owned references and immediately reuse GPU KV.
    # Only the staging task's checkpoint reference must remain until completion.
    for page in pages:
        page.checkpoint_pool.free_state_cache([page.checkpoint_id])
        assert page.checkpoint_pool.refs[page.checkpoint_id] == 1
    for buffer in manager.kv_buffer:
        buffer.zero_()
    cpu_stream.synchronize()
    obj.update_cpu_cache_task_states()
    assert obj.dsv4_sessions == {} and client.ready == {1, 2, 4}
    for page, layer_values, (_, _, state_value) in zip(pages, expected, [(2048, 2, 11), (768, 4, 22), (1280, 1, 33)]):
        offset = 0
        cpu_page = client.cpu_kv_cache_tensor[page.cpu_page_index]
        for data, slot_bytes in zip(layer_values, manager.cache_config.compressed_kv_slot_bytes):
            torch.testing.assert_close(cpu_page[offset : offset + data.numel()], data, rtol=0, atol=0)
            offset += page_tokens * slot_bytes
        assert cpu_page[offset:].eq(state_value).all()
        assert page.checkpoint_pool.refs[page.checkpoint_id] == 0
