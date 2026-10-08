from types import SimpleNamespace
from collections import deque

import pytest
import torch

from lightllm.server.multi_level_kv_cache import (
    AdaptiveCachePlacementController,
    CacheCapacityConfig,
    CacheTier,
)
from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule
from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as multi_level_kv_cache_impl
from lightllm.server.router.model_infer.infer_batch import InferenceContext


def test_cache_tiers_reassignment_is_rejected():
    context = InferenceContext()
    context.cache_placement_controller = AdaptiveCachePlacementController(
        capacity=CacheCapacityConfig(1, 1, 1),
        args=SimpleNamespace(enable_cpu_cache=True, enable_disk_cache=True),
    )
    reqs = [
        SimpleNamespace(
            cache_tiers=(CacheTier.GPU,),
            shm_req=SimpleNamespace(group_req_id=index, request_id=index, input_len=length),
        )
        for index, length in enumerate((100, 200, 300))
    ]

    context.cache_placement_controller.set_req_cache_way(reqs)
    context.cache_placement_controller = AdaptiveCachePlacementController(
        capacity=CacheCapacityConfig(0, 1, 0),
        args=SimpleNamespace(enable_cpu_cache=True, enable_disk_cache=False),
    )

    with pytest.raises(AssertionError):
        context.cache_placement_controller.set_req_cache_way(reqs)


def test_non_gpu_cache_tiers_release_owned_tokens_without_radix_insert():
    released_refs = []
    context = InferenceContext()
    context.is_hybrid_att_model = False
    context.req_manager = SimpleNamespace(req_to_token_indexs=torch.tensor([[10, 11, 12, 13, 14, 15, 16]]))
    context.radix_cache = SimpleNamespace(dec_node_ref_counter=released_refs.append)
    shared_node = SimpleNamespace(node_prefix_total_len=2)
    req = SimpleNamespace(req_idx=0, cur_kv_len=5, hold_kv_len=7, shared_kv_node=shared_node)
    free_token_indexes = []

    context._free_req_mem_without_radix_insert(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == [12, 13, 14, 15, 16]
    assert released_refs == [shared_node]
    assert req.shared_kv_node is None


def test_legacy_cache_tiers_still_insert_gpu_radix_cache():
    context = InferenceContext()
    context.radix_cache = object()
    context.is_hybrid_att_model = False
    inserted_reqs = []
    context._full_att_free_req = lambda free_token_index, req: inserted_reqs.append(req)
    req = SimpleNamespace(
        cache_tiers=(CacheTier.GPU, CacheTier.CPU, CacheTier.DISK),
        cur_kv_len=3,
        shm_req=SimpleNamespace(shm_cur_kv_len=3),
    )

    context.free_a_req_mem([], req)

    assert inserted_reqs == [req]


def test_finished_batch_routes_cpu_and_disk_offloads_separately(monkeypatch):
    class NotStartedStatus:
        @staticmethod
        def is_finished():
            return False

        @staticmethod
        def is_running():
            return False

        @staticmethod
        def is_not_started():
            return True

    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.args = SimpleNamespace(cpu_cache_token_page_size=64, linear_att_hash_page_size=64)
    module.backend = SimpleNamespace(radix_cache=object())
    module.cpu_cache_handle_queue = deque()
    module.need_sync_compute_stream = lambda: False
    offload_calls = []

    def start_offload(req, cpu_kv_cache_stream):
        offload_calls.append((req.shm_req.request_id, CacheTier.DISK in req.cache_tiers))
        return SimpleNamespace(req=req)

    module._start_kv_cache_offload_task = start_offload
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "is_hybrid_att_model", False)
    monkeypatch.setattr(
        multi_level_kv_cache_impl.g_infer_context,
        "get_cpu_kv_cache_stream",
        lambda: object(),
    )
    cache_tiers = (
        (CacheTier.GPU,),
        (CacheTier.CPU,),
        (CacheTier.CPU, CacheTier.DISK),
    )
    reqs = [
        SimpleNamespace(
            cache_tiers=req_cache_tiers,
            cur_kv_len=length,
            cpu_cache_task_status=NotStartedStatus(),
            shm_req=SimpleNamespace(
                group_req_id=index,
                request_id=index,
                input_len=length,
            ),
        )
        for index, (length, req_cache_tiers) in enumerate(zip((100, 200, 300), cache_tiers))
    ]
    offload_reqs = [req for req in reqs if CacheTier.CPU in req.cache_tiers or CacheTier.DISK in req.cache_tiers]

    true_finished_reqs = module.offload_finished_reqs_to_cpu_cache(offload_reqs)

    assert true_finished_reqs == []
    assert offload_calls == [(1, False), (2, True)]
    assert len(module.cpu_cache_handle_queue) == 2


def test_non_gpu_linear_cache_tiers_release_pending_state_pages():
    freed_small_pages = []
    freed_big_pages = []
    context = InferenceContext()
    context.is_hybrid_att_model = True
    context.req_manager = SimpleNamespace(req_to_token_indexs=torch.tensor([[10, 11, 12]]))
    context.radix_cache = SimpleNamespace(
        small_page_buffers=SimpleNamespace(free_state_cache=freed_small_pages.extend),
        big_page_buffers=SimpleNamespace(free_state_cache=freed_big_pages.extend),
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=3,
        hold_kv_len=3,
        shared_kv_node=None,
        tail_small_page_buffer_id=7,
        hybrid_len_to_big_page_id={128: 8, 256: 9},
    )
    free_token_indexes = []

    context._free_req_mem_without_radix_insert(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == [10, 11, 12]
    assert freed_small_pages == [7]
    assert freed_big_pages == [8, 9]
    assert req.tail_small_page_buffer_id is None
    assert req.hybrid_len_to_big_page_id == {}


def test_cpu_cache_load_uses_exact_aligned_size(monkeypatch):
    table = torch.full((1, 136), -1, dtype=torch.int32)
    table[0, :4] = torch.arange(4, dtype=torch.int32)
    next_index = 4
    alloc_sizes = []
    loaded_indexes = []

    def alloc(need_size):
        nonlocal next_index
        alloc_sizes.append(need_size)
        mem_indexes = torch.arange(next_index, next_index + need_size, dtype=torch.int32)
        next_index += need_size
        return mem_indexes

    def alloc_req_kv_mem(req, alloc_token_num):
        mem_indexes = alloc(alloc_token_num)
        new_hold_kv_len = req.hold_kv_len + alloc_token_num
        table[req.req_idx, req.hold_kv_len : new_hold_kv_len] = mem_indexes
        req.hold_kv_len = new_hold_kv_len
        return mem_indexes

    operator = SimpleNamespace(
        load_cpu_cache_to_gpu=lambda mem_indexes, **kwargs: loaded_indexes.extend(mem_indexes.tolist())
    )
    backend = SimpleNamespace(
        is_master_in_dp=False,
        radix_cache=None,
        model=SimpleNamespace(
            mem_manager=SimpleNamespace(operator=operator),
            req_manager=SimpleNamespace(req_to_token_indexs=table),
        ),
        _alloc_req_kv_mem=alloc_req_kv_mem,
    )
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.args = SimpleNamespace(page_size=4, cpu_cache_token_page_size=132)
    module.backend = backend
    module.init_sync_group = object()
    module.need_sync_compute_stream = lambda: False
    module.cpu_cache_client = SimpleNamespace()
    context = SimpleNamespace(
        is_deepseek_v4=False,
        req_manager=SimpleNamespace(req_to_token_indexs=table, mem_manager=SimpleNamespace(alloc=alloc)),
        get_can_alloc_token_num=lambda: 200,
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=4,
        hold_kv_len=4,
        shm_req=SimpleNamespace(
            input_len=256,
            disk_prompt_cache_len=0,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [7]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [132]),
        ),
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
    )
    monkeypatch.setattr(multi_level_kv_cache_impl, "g_infer_context", context)
    monkeypatch.setattr(multi_level_kv_cache_impl.dist, "barrier", lambda group: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)

    module.load_cpu_cache_to_reqs([req])

    assert alloc_sizes == [128]
    assert req.cur_kv_len == 132
    assert req.hold_kv_len == 132
    assert loaded_indexes == list(range(132))
    assert table[0, :132].tolist() == list(range(132))


@pytest.mark.parametrize("disk_cache", [False, True])
def test_dsv4_store_survives_request_finish_and_waits_for_load(disk_cache, monkeypatch):
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    slot = SimpleNamespace(in_use=True)
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        model=SimpleNamespace(mem_manager=SimpleNamespace(operator=SimpleNamespace(cpu_cache_staging_slots=[slot]))),
    )
    module.args = SimpleNamespace(enable_disk_cache=disk_cache, cpu_cache_token_page_size=2048)
    load_event = SimpleNamespace(ready=False)
    load_event.query = lambda: load_event.ready
    store_event = SimpleNamespace(ready=False)
    store_event.query = lambda: store_event.ready
    session = multi_level_kv_cache_impl.Dsv4CpuStoreSession(
        request_id=3, leased_pages=[5, 7], pending_task_num=1, load_submitted=True, load_event=load_event
    )
    module._dsv4_store_sessions = {3: session}
    module._dsv4_store_tasks = deque(
        [multi_level_kv_cache_impl.Dsv4StoreTask([7], [session], 0, object(), store_event)]
    )
    published = []
    released = []
    module.cpu_cache_client = SimpleNamespace(
        lock=SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
        update_pages_status_to_ready=lambda page_list, **kwargs: published.append((list(page_list), kwargs)),
        deref_pages=lambda pages: released.extend(pages),
        check_allpages_ready=lambda pages: True,
    )
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "is_deepseek_v4", True)
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "infer_req_ids", [])
    req = SimpleNamespace(req_id=3)

    assert module.offload_finished_reqs_to_cpu_cache([req]) == [req]
    assert session.closing and session.pending_task_num == 1
    assert published == released == []
    assert slot.in_use

    store_event.ready = True
    module.update_cpu_cache_task_states()
    assert published == [([7], {"deref": False})]
    assert released == [] and 3 in module._dsv4_store_sessions
    assert session.pending_task_num == 0 and not slot.in_use

    load_event.ready = True
    module.update_cpu_cache_task_states()
    assert module._dsv4_store_sessions == {}
    if disk_cache:
        assert published[-1] == ([5, 7], {"deref": True, "disk_offload_enable": True, "token_num_in_page_list": 4096})
    else:
        assert released == [7, 5]


@pytest.mark.parametrize("is_deepseek_v4", [False, True])
@pytest.mark.parametrize("is_master_in_dp", [False, True])
@pytest.mark.parametrize("mixed_batch", [False, True])
def test_prompt_logprobs_filter_is_shared_by_models(is_deepseek_v4, is_master_in_dp, mixed_batch, monkeypatch):
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.backend = SimpleNamespace(is_master_in_dp=is_master_in_dp)
    loaded = []
    released = []
    module._load_dsv4_cpu_cache_to_reqs = loaded.extend
    module._load_standard_cpu_cache_to_reqs = loaded.extend
    module.cpu_cache_client = SimpleNamespace(
        lock=SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
        deref_pages=released.extend,
    )
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "is_deepseek_v4", is_deepseek_v4)
    req = SimpleNamespace(
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=0)),
        shm_req=SimpleNamespace(
            cpu_prompt_cache_len=2048,
            disk_prompt_cache_len=2048,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [4, 5]),
        ),
    )
    cache_req = SimpleNamespace(
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        shm_req=SimpleNamespace(cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [7])),
    )
    module.load_cpu_cache_to_reqs([req, cache_req] if mixed_batch else [req])
    assert loaded == ([cache_req] if mixed_batch else [])
    expected_pages = [4, 5, 7] if mixed_batch and not is_deepseek_v4 else [4, 5]
    assert released == (expected_pages if is_master_in_dp else [])
    assert req.shm_req.cpu_prompt_cache_len == req.shm_req.disk_prompt_cache_len == (0 if is_master_in_dp else 2048)


def test_standard_load_releases_skipped_and_matched_pages_once_after_barrier(monkeypatch):
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.backend = SimpleNamespace(is_master_in_dp=True)
    module.init_sync_group = object()
    events = []
    module.cpu_cache_client = SimpleNamespace(
        lock=SimpleNamespace(
            acquire_sleep1ms=lambda: events.append("lock"),
            release=lambda: events.append("unlock"),
        ),
        deref_pages=lambda pages: events.append(("deref", list(pages))),
    )
    reqs = [
        SimpleNamespace(
            cur_kv_len=0,
            sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=logprobs)),
            shm_req=SimpleNamespace(
                input_len=64,
                disk_prompt_cache_len=0,
                cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda page=page: [page]),
                token_hash_page_len_list=SimpleNamespace(get_all=lambda: [64]),
            ),
        )
        for page, logprobs in ((4, 0), (7, -1))
    ]
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "is_deepseek_v4", False)
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "get_can_alloc_token_num", lambda: 128)
    monkeypatch.setattr(multi_level_kv_cache_impl.dist, "barrier", lambda group: events.append("barrier"))

    module.load_cpu_cache_to_reqs(reqs)

    assert events == ["barrier", "lock", ("deref", [4, 7]), "unlock"]
