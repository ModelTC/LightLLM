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
    context.is_linear_att_mixed_model = False
    context.req_manager = SimpleNamespace(req_to_token_indexs=torch.tensor([[10, 11, 12, 13, 14]]))
    context.radix_cache = SimpleNamespace(dec_node_ref_counter=released_refs.append)
    shared_node = SimpleNamespace(node_prefix_total_len=2)
    req = SimpleNamespace(req_idx=0, cur_kv_len=5, shared_kv_node=shared_node)
    free_token_indexes = []

    context._free_req_mem_without_radix_insert(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == [12, 13, 14]
    assert released_refs == [shared_node]
    assert req.shared_kv_node is None


def test_legacy_cache_tiers_still_insert_gpu_radix_cache():
    context = InferenceContext()
    context.radix_cache = object()
    context.is_linear_att_mixed_model = False
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
    monkeypatch.setattr(multi_level_kv_cache_impl.g_infer_context, "is_linear_att_mixed_model", False)
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
    context.is_linear_att_mixed_model = True
    context.req_manager = SimpleNamespace(req_to_token_indexs=torch.tensor([[10, 11, 12]]))
    context.radix_cache = SimpleNamespace(
        linear_att_small_page_buffers=SimpleNamespace(free_state_cache=freed_small_pages.extend),
        linear_att_big_page_buffers=SimpleNamespace(free_state_cache=freed_big_pages.extend),
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=3,
        shared_kv_node=None,
        tail_linear_att_small_page_buffer_id=7,
        linear_att_len_to_big_page_id={128: 8, 256: 9},
    )
    free_token_indexes = []

    context._free_req_mem_without_radix_insert(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == [10, 11, 12]
    assert freed_small_pages == [7]
    assert freed_big_pages == [8, 9]
    assert req.tail_linear_att_small_page_buffer_id is None
    assert req.linear_att_len_to_big_page_id == {}
