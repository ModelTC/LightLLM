from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.multi_level_kv_cache import AdaptiveCachePlacementController, CacheCapacityConfig, CacheTier
from lightllm.server.router.model_infer.infer_batch import InferenceContext
from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as multi_level_kv_cache_impl
from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule


def test_running_cpu_cache_tasks_are_not_ready(monkeypatch):
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend import base_backend as base_backend_impl
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend

    status = InferReq._CpuCacheTaskStatus

    def req(load_status, offload_status):
        return SimpleNamespace(
            cpu_cache_load_task_status=load_status,
            cpu_cache_offload_task_status=offload_status,
        )

    running_load = req(status.RUNNING, status.NOT_STARTED)
    running_offload = req(status.FINISHED, status.RUNNING)
    ready = req(status.NOT_STARTED, status.NOT_STARTED)
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = SimpleNamespace(enable_cpu_cache=True)
    monkeypatch.setattr(
        base_backend_impl,
        "g_infer_context",
        SimpleNamespace(requests_mapping={1: running_load, 2: running_offload, 3: ready}),
    )

    assert backend._filter_not_ready_reqs([1, 2, 3]) == [ready]
    backend.args.enable_cpu_cache = False
    assert backend._filter_not_ready_reqs([1, 2, 3]) == [running_load, running_offload, ready]


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
    context.req_manager = SimpleNamespace()
    context.radix_cache = SimpleNamespace(dec_node_ref_counter=released_refs.append)
    shared_node = SimpleNamespace(node_prefix_total_len=2)
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=5,
        hold_kv_len=7,
        shared_kv_node=shared_node,
        token_index=SimpleNamespace(
            get_free_token_indexes=lambda start, end: torch.arange(10 + start, 10 + end),
            get_cpu_token_indexes=lambda start, end: torch.arange(10 + start, 10 + end),
            reset=lambda: None,
        ),
    )
    free_token_index_blocks = []

    context._free_req_mem_without_radix_insert(free_token_index_blocks, req)

    assert free_token_index_blocks[0].tolist() == [12, 13, 14, 15, 16]
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
        token_index=SimpleNamespace(reset=lambda: None),
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
    module.offload_tasks = deque()
    module.need_sync_compute_stream = lambda: False
    offload_calls = []

    def start_offload(req):
        offload_calls.append((req.shm_req.request_id, CacheTier.DISK in req.cache_tiers))
        return SimpleNamespace(req=req)

    module.write_req_to_cpu = start_offload
    current = torch.cuda.current_stream()
    offload_stream = torch.cuda.Stream()
    waited = []
    offload_stream.wait_stream = lambda stream: waited.append(stream)
    module.get_cpu_offload_stream = lambda: offload_stream
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "is_hybrid_att_model", False)
    cache_tiers = (
        (CacheTier.GPU,),
        (CacheTier.CPU,),
        (CacheTier.CPU, CacheTier.DISK),
    )
    reqs = [
        SimpleNamespace(
            cache_tiers=req_cache_tiers,
            cur_kv_len=length,
            cpu_cache_offload_task_status=NotStartedStatus(),
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

    assert waited == [current]
    assert true_finished_reqs == []
    assert offload_calls == [(1, False), (2, True)]
    assert len(module.offload_tasks) == 2

    overlap = torch.cuda.Stream()
    waited_by_overlap = []
    overlap.wait_stream = lambda stream: waited_by_overlap.append(stream)
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "get_overlap_stream", lambda: overlap)
    module.need_sync_compute_stream = lambda: True
    waited.clear()
    assert module.offload_finished_reqs_to_cpu_cache([]) == []
    assert waited == [] and waited_by_overlap == []
    short_req = SimpleNamespace(
        cache_tiers=(CacheTier.CPU,),
        cur_kv_len=1,
        cpu_cache_offload_task_status=NotStartedStatus(),
        shm_req=SimpleNamespace(group_req_id=0, request_id=0, input_len=1),
    )
    assert module.offload_finished_reqs_to_cpu_cache([short_req]) == [short_req]
    assert waited == [current, overlap]
    assert waited_by_overlap == [offload_stream]


def test_non_gpu_linear_cache_tiers_release_pending_state_pages():
    freed_small_pages = []
    freed_big_pages = []
    context = InferenceContext()
    context.is_hybrid_att_model = True
    context.req_manager = SimpleNamespace()
    context.att_state_manager = SimpleNamespace(
        bounded_att_small_page_allocator=SimpleNamespace(free=freed_small_pages.extend),
        bounded_att_big_page_allocator=SimpleNamespace(free=freed_big_pages.extend),
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=3,
        hold_kv_len=3,
        token_index=SimpleNamespace(
            get_free_token_indexes=lambda start, end: torch.arange(10 + start, 10 + end),
            reset=lambda: None,
        ),
        shared_kv_node=None,
        tail_small_page_buffer_id=7,
        hybrid_len_to_big_page_id={128: 8, 256: 9},
    )
    free_token_index_blocks = []

    context._free_req_mem_without_radix_insert(free_token_index_blocks, req)

    assert free_token_index_blocks[0].tolist() == [10, 11, 12]
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

    def alloc_token_indexes(req, end, allocator, page_size, radix_cache=None):
        alloc_token_num = end - req.hold_kv_len
        mem_indexes = alloc(alloc_token_num)
        new_hold_kv_len = req.hold_kv_len + alloc_token_num
        table[req.req_idx, req.hold_kv_len : new_hold_kv_len] = mem_indexes
        req.hold_kv_len = new_hold_kv_len
        return mem_indexes

    resources = SimpleNamespace(
        get_cpu_cache_operator=lambda: SimpleNamespace(
            get_page_layouts=lambda: {},
            read_req=lambda cpu_page_buffers, cpu_page_indices, token_indices, **kwargs: loaded_indexes.extend(
                token_indices.tolist()
            ),
        ),
        full_att_allocator=None,
    )
    backend = SimpleNamespace(
        is_master_in_dp=False,
        dp_world_size=1,
        is_hybrid_att_model=False,
        radix_cache=None,
        model=SimpleNamespace(
            att_state_manager=resources,
            req_manager=SimpleNamespace(
                req_to_token_indexs=table, alloc_token_indexes=alloc_token_indexes, wait_token_indexes=lambda: None
            ),
        ),
    )
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.load_tasks = deque()
    module.args = SimpleNamespace(page_size=4, cpu_cache_token_page_size=132)
    module.backend = backend
    module.init_sync_group = object()
    module.need_sync_compute_stream = lambda: False
    module.cpu_cache_client = SimpleNamespace(cpu_kv_cache_tensor=None)
    context = SimpleNamespace(
        req_manager=SimpleNamespace(req_to_token_indexs=table, alloc_token_indexes=alloc_token_indexes),
        radix_cache=None,
        get_can_alloc_token_num=lambda: 200,
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=4,
        hold_kv_len=4,
        mm_input_spans=[],
        shm_req=SimpleNamespace(
            input_len=256,
            disk_prompt_cache_len=0,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [7]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [132]),
        ),
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        args=SimpleNamespace(page_size=4),
    )
    req.token_index = SimpleNamespace(get_gpu_token_indexes=lambda start, end: table[req.req_idx, start:end])
    req.kv_cache_alloc_need = lambda target: max(0, (target + 3) // 4 * 4 - req.hold_kv_len)
    req.alloc_kv_mem = lambda alloc_token_num: alloc_token_indexes(req, req.hold_kv_len + alloc_token_num, None, 4)
    monkeypatch.setattr(multi_level_kv_cache_impl, "g_infer_context", context)
    monkeypatch.setattr(multi_level_kv_cache_impl.dist, "barrier", lambda group: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)

    module.load_cpu_cache_to_reqs([req])

    assert alloc_sizes == [128]
    assert req.cur_kv_len == 132
    assert req.hold_kv_len == 132
    assert loaded_indexes == list(range(132))
    assert table[0, :132].tolist() == list(range(132))


def test_standard_load_releases_unloaded_pages_immediately(monkeypatch):
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.backend = SimpleNamespace(is_master_in_dp=True, dp_world_size=1)
    module.init_sync_group = object()
    module.need_sync_compute_stream = lambda: False
    events = []
    module.cpu_cache_client = SimpleNamespace(
        lock=SimpleNamespace(
            acquire_sleep1ms=lambda: events.append("lock"),
            release=lambda: events.append("unlock"),
        ),
        deref_pages=lambda page_list: events.append(("deref", list(page_list))),
    )
    reqs = [
        SimpleNamespace(
            cur_kv_len=0,
            mm_input_spans=[],
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
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "get_can_alloc_token_num", lambda: 128)

    module.load_cpu_cache_to_reqs(reqs)

    assert events == ["lock", ("deref", [4]), ("deref", [7]), "unlock"]
    assert not getattr(module, "load_tasks", None)
    assert all(req.cpu_cache_load_task_status.is_finished() for req in reqs)


def _load_module(monkeypatch, *, sync, master=True, world=1):
    events = []

    def alloc_token_indexes(req, end, allocator, page_size, radix_cache=None):
        req.hold_kv_len = end
        return torch.arange(req.cur_kv_len, end, dtype=torch.int32)

    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.args = SimpleNamespace(page_size=4, cpu_cache_token_page_size=128)
    module.backend = SimpleNamespace(
        is_master_in_dp=master,
        dp_world_size=world,
        is_hybrid_att_model=False,
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                req_to_token_indexs=torch.arange(512, dtype=torch.int32).view(1, -1),
                alloc_token_indexes=alloc_token_indexes,
                wait_token_indexes=lambda: None,
            ),
            att_state_manager=SimpleNamespace(
                get_cpu_cache_operator=lambda: SimpleNamespace(
                    get_page_layouts=lambda: {},
                    read_req=lambda **kwargs: events.append("copy"),
                ),
                full_att_allocator=None,
            ),
        ),
    )
    module.need_sync_compute_stream = lambda: sync
    module.init_sync_group = module.filter_group = object()
    module.load_sync_tensor = object()
    module.load_tasks = deque()
    module.cpu_cache_client = SimpleNamespace(
        cpu_kv_cache_tensor=None,
        lock=SimpleNamespace(acquire_sleep1ms=lambda: events.append("lock"), release=lambda: events.append("unlock")),
        deref_pages=lambda page_list: events.append(("deref", list(page_list))),
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=0,
        hold_kv_len=0,
        mm_input_spans=[],
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        shm_req=SimpleNamespace(
            input_len=256,
            disk_prompt_cache_len=0,
            cpu_prompt_cache_len=0,
            shm_cur_kv_len=0,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: [3, 4]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [128, 256]),
        ),
        args=SimpleNamespace(page_size=4),
    )
    token_table = module.backend.model.req_manager.req_to_token_indexs
    req.token_index = SimpleNamespace(get_gpu_token_indexes=lambda start, end: token_table[req.req_idx, start:end])
    req.kv_cache_alloc_need = lambda target: max(0, (target + 3) // 4 * 4 - req.hold_kv_len)
    req.alloc_kv_mem = lambda alloc_token_num: alloc_token_indexes(req, req.hold_kv_len + alloc_token_num, None, 4)
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "get_can_alloc_token_num", lambda: 512)
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "get_overlap_stream", torch.cuda.Stream)
    monkeypatch.setattr(multi_level_kv_cache_impl.dist, "barrier", lambda group: events.append("barrier"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    return module, req, events


def test_async_load_defers_deref_until_the_event_is_notified(monkeypatch):
    module, req, events = _load_module(monkeypatch, sync=False)
    current = torch.cuda.current_stream()
    overlap = torch.cuda.Stream()
    waited_by_load = []
    overlap.wait_stream = lambda stream: events.append("wait")
    load_stream = module.get_cpu_load_stream()
    load_stream.wait_stream = lambda stream: waited_by_load.append(stream)
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "get_overlap_stream", lambda: overlap)
    recorded = []

    class Event:
        def __init__(self):
            self.ready = False

        def record(self):
            recorded.append(self)

        def query(self):
            return self.ready

    monkeypatch.setattr(multi_level_kv_cache_impl.torch.cuda, "Event", Event)
    module.load_cpu_cache_to_reqs([req])
    assert waited_by_load == [current]
    assert events == ["copy"] and len(recorded) == 1 and len(module.load_tasks) == 1
    module.update_cpu_cache_load_task_states()
    assert "deref" not in [item[0] for item in events if isinstance(item, tuple)]
    recorded[0].ready = True
    module.update_cpu_cache_load_task_states()
    assert events[-3:] == ["lock", ("deref", [3, 4]), "unlock"] and not module.load_tasks


def test_synchronous_backend_waits_overlap_for_the_load_stream(monkeypatch):
    module, req, events = _load_module(monkeypatch, sync=True)
    current = torch.cuda.current_stream()
    overlap = torch.cuda.Stream()
    waited_by_overlap = []
    waited_by_load = []
    overlap.wait_stream = lambda stream: waited_by_overlap.append(stream)
    load_stream = module.get_cpu_load_stream()
    load_stream.wait_stream = lambda stream: waited_by_load.append(stream)
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "get_overlap_stream", lambda: overlap)
    module.load_cpu_cache_to_reqs([req])
    assert waited_by_load == [current, overlap]
    assert waited_by_overlap == [load_stream]
    assert events == ["copy"] and len(module.load_tasks) == 1
    module.load_tasks[0].event.synchronize()
    module.update_cpu_cache_load_task_states()
    assert events[-3:] == ["lock", ("deref", [3, 4]), "unlock"] and not module.load_tasks


def test_async_load_notifies_every_rank_before_deref(monkeypatch):
    module, req, events = _load_module(monkeypatch, sync=False, world=2)
    collectives = []

    class Event:
        def __init__(self):
            self.ready = True

        def record(self):
            return None

        def query(self):
            return self.ready

    monkeypatch.setattr(multi_level_kv_cache_impl.torch.cuda, "Event", Event)
    monkeypatch.setattr(
        multi_level_kv_cache_impl.dist,
        "all_reduce",
        lambda tensor, group: collectives.append(("all_reduce", group)),
    )
    monkeypatch.setattr(
        multi_level_kv_cache_impl.dist,
        "broadcast_object_list",
        lambda data, group, group_src: collectives.append(("broadcast", data[0], group)),
    )
    module.load_cpu_cache_to_reqs([req])
    assert collectives == [("all_reduce", module.init_sync_group)] and not any(
        isinstance(item, tuple) and item[0] == "deref" for item in events
    )
    module.update_cpu_cache_load_task_states()
    assert collectives[1][0] == "broadcast" and collectives[1][1] == 1
    assert ("deref", [3, 4]) in events


def test_fa3_cpu_cache_sync_defaults_off():
    from lightllm.common.attention.backends.base import BaseAttentionBackend
    from lightllm.common.attention.backends.dense.fa3.backend import Fa3AttBackend
    from lightllm.common.attention.backends.dense.fa3.fp8 import Fp8Fa3AttBackend
    from lightllm.common.attention.backends.dense.fa3.neo import NeoFa3AttBackend
    from lightllm.common.attention.backends.mla.fa3.backend import MlaFa3AttBackend

    assert BaseAttentionBackend.requires_cpu_cache_sync is False
    assert all(
        backend.requires_cpu_cache_sync is False
        for backend in (Fa3AttBackend, Fp8Fa3AttBackend, MlaFa3AttBackend, NeoFa3AttBackend)
    )
