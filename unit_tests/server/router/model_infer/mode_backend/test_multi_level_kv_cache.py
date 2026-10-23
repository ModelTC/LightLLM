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
from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext
from unit_tests.common.cpu_index_helpers import make_cpu_index_manager, bind_cpu_indexes
from lightllm.server.router.model_infer.infer_req_ext1 import PinMemPageCacheReqTokenIndex, create_req_token_index


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
    context.req_manager = make_cpu_index_manager(token_table=torch.tensor([[10, 11, 12, 13, 14, 15, 16]]), page_size=1)
    context.radix_cache = SimpleNamespace(dec_node_ref_counter=released_refs.append)
    shared_node = SimpleNamespace(node_prefix_total_len=2)
    req = SimpleNamespace(req_idx=0, cur_kv_len=5, hold_kv_len=7, shared_kv_node=shared_node)
    bind_cpu_indexes(req, context.req_manager)
    free_token_indexes = []

    context._free_req_mem_without_radix_insert(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == [12, 13, 14, 15, 16]
    assert released_refs == [shared_node]
    assert req.shared_kv_node is None


def test_legacy_cache_tiers_still_insert_gpu_radix_cache():
    context = InferenceContext()
    context.radix_cache = object()
    context.is_hybrid_att_model = False
    context.req_manager = make_cpu_index_manager(token_table=torch.full((1, 4), -1, dtype=torch.int32))
    inserted_reqs = []
    context._full_att_free_req = lambda free_token_index, req: inserted_reqs.append(req)
    req = SimpleNamespace(
        req_idx=0,
        cache_tiers=(CacheTier.GPU, CacheTier.CPU, CacheTier.DISK),
        cur_kv_len=3,
        shm_req=SimpleNamespace(shm_cur_kv_len=3),
    )

    bind_cpu_indexes(req, context.req_manager)

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
    context.req_manager = make_cpu_index_manager(token_table=torch.tensor([[10, 11, 12]]), page_size=1)
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
    bind_cpu_indexes(req, context.req_manager)
    free_token_indexes = []

    context._free_req_mem_without_radix_insert(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == [10, 11, 12]
    assert freed_small_pages == [7]
    assert freed_big_pages == [8, 9]
    assert req.tail_small_page_buffer_id is None
    assert req.hybrid_len_to_big_page_id == {}


@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned CPU buffers")
def test_cpu_cache_load_uses_exact_aligned_size(monkeypatch, mode):
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_page_offsets", None)
    table = torch.full((1, 136), -1, dtype=torch.int32)
    gpu_table = torch.full_like(table, -2)
    table[0, :4] = torch.arange(4, dtype=torch.int32)
    next_index = 4
    alloc_sizes = []
    loaded_indexes = []

    def alloc(need_size):
        nonlocal next_index
        alloc_sizes.append(need_size)
        mem_indexes = torch.arange(next_index, next_index + need_size, dtype=torch.int32, pin_memory=True)
        next_index += need_size
        return mem_indexes

    def alloc_kv_mem(alloc_token_num):
        mem_indexes = alloc(alloc_token_num)
        new_hold_kv_len = req.hold_kv_len + alloc_token_num
        req.token_index.append_token_indexes(mem_indexes)
        table[req.req_idx, req.hold_kv_len : new_hold_kv_len] = mem_indexes
        req.hold_kv_len = new_hold_kv_len
        return mem_indexes

    def load(mem_indexes, **kwargs):
        assert mem_indexes.data_ptr() == gpu_table[0, :132].data_ptr()
        loaded_indexes.extend(mem_indexes.tolist())

    operator = SimpleNamespace(load_cpu_cache_to_gpu=load)
    backend = SimpleNamespace(
        is_master_in_dp=False,
        radix_cache=None,
        model=SimpleNamespace(
            mem_manager=SimpleNamespace(operator=operator),
            req_manager=make_cpu_index_manager(token_table=table, req_to_token_indexs=gpu_table, page_size=4),
        ),
    )
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.args = SimpleNamespace(page_size=4, cpu_cache_token_page_size=132)
    module.backend = backend
    module.init_sync_group = object()
    module.need_sync_compute_stream = lambda: False
    module.cpu_cache_client = SimpleNamespace()
    context = SimpleNamespace(
        req_manager=make_cpu_index_manager(token_table=table, mem_manager=SimpleNamespace(alloc=alloc), page_size=1),
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
        alloc_kv_mem=alloc_kv_mem,
    )
    bind_cpu_indexes(req, backend.model.req_manager)
    req.args.req_token_index_mode = mode
    req.token_index = create_req_token_index(req, index_stream=torch.cuda.default_stream())
    req.token_index.append_token_indexes(table[0, :4].pin_memory())
    backend.model.req_manager._req_token_indexes[0] = req.token_index
    context.req_manager = backend.model.req_manager
    monkeypatch.setattr(multi_level_kv_cache_impl, "g_infer_context", context)
    monkeypatch.setattr(multi_level_kv_cache_impl.dist, "barrier", lambda group: None)
    uploaded_indexes = []

    def upload(tensor, **kwargs):
        uploaded_indexes.append(tensor)
        return tensor

    def unexpected_copy(*args, **kwargs):
        pytest.fail("CPU page-index upload must not concatenate GPU tensors")

    monkeypatch.setattr(torch.Tensor, "cuda", upload)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(torch, "cat", unexpected_copy)

    module.load_cpu_cache_to_reqs([req])

    assert len(uploaded_indexes) == 1  # 仅页面编号使用 .cuda()；索引上传到已有 GPU 表。
    assert alloc_sizes == [128]
    assert req.cur_kv_len == 132
    assert req.hold_kv_len == 132
    assert loaded_indexes == list(range(132))
    assert table[0, :132].tolist() == list(range(132))
    assert gpu_table[0, :132].tolist() == list(range(132))
    assert gpu_table[0, 132:].eq(-2).all()


@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("use_default_index_stream", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_cache_offload_getter_orders_transfer_stream(monkeypatch, mode, use_default_index_stream):
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_page_offsets", None)
    default_stream = torch.cuda.default_stream()
    index_stream = default_stream if use_default_index_stream else torch.cuda.Stream()
    index_stream.wait_stream(default_stream)
    transfer_stream = torch.cuda.Stream()
    gpu_indexes = torch.full((64,), -1, dtype=torch.int32, device="cuda")
    req = SimpleNamespace(
        req_idx=0,
        req_manager=SimpleNamespace(req_to_token_indexs=gpu_indexes.unsqueeze(0), page_size=64),
        args=SimpleNamespace(req_token_index_mode=mode),
        cache_tiers=(CacheTier.CPU,),
        cur_kv_len=64,
        shm_req=SimpleNamespace(
            input_len=64,
            token_hash_list=SimpleNamespace(get_all=lambda: [123]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [64]),
        ),
    )
    helper = create_req_token_index(req, index_stream=index_stream)
    req.token_index = helper
    expected = torch.arange(64, dtype=torch.int32, pin_memory=True)
    # 延迟索引流上的写入，验证 getter 在卸载流调用时也会承接依赖。
    with torch.cuda.stream(index_stream):
        torch.cuda._sleep(20_000_000)
    helper.append_token_indexes(expected)
    order = []
    observed_indexes = []
    original_get_gpu_token_indexes = helper.get_gpu_token_indexes

    def prepare_indexes(start, end):
        assert torch.cuda.current_stream() == transfer_stream
        order.append("prepare")
        return original_get_gpu_token_indexes(start, end)

    original_wait_stream = torch.cuda.Stream.wait_stream

    def wait_stream(waiting_stream, source_stream):
        assert waiting_stream == transfer_stream and source_stream == index_stream
        order.append("transfer_waits_index")
        original_wait_stream(waiting_stream, source_stream)

    def offload(mem_indexes, **kwargs):
        assert torch.cuda.current_stream() == transfer_stream
        order.append("offload")
        observed_indexes.append(mem_indexes.clone())

    def broadcast(data, **kwargs):
        data[0] = {"item_size": 1, "page_list": [3], "ready_list": [False]}

    req.token_index.get_gpu_token_indexes = prepare_indexes
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.backend = SimpleNamespace(
        is_master_in_dp=False,
        dp_world_size=1,
        model=SimpleNamespace(
            mem_manager=SimpleNamespace(operator=SimpleNamespace(offload_gpu_kv_to_cpu_cache=offload))
        ),
    )
    module.gloo_group = object()
    module.cpu_cache_client = object()
    module.page_index_buffer = torch.empty(8, dtype=torch.int32, device="cuda")
    module.page_ready_buffer = torch.empty(8, dtype=torch.bool, device="cuda")
    monkeypatch.setattr(torch.cuda.Stream, "wait_stream", wait_stream)
    monkeypatch.setattr(multi_level_kv_cache_impl.dist, "broadcast_object_list", broadcast)

    task = module._start_kv_cache_offload_task(req, transfer_stream)
    task.sync_event.synchronize()

    assert order == ["prepare", "transfer_waits_index", "offload"]
    assert torch.cuda.current_stream() == default_stream
    assert torch.equal(observed_indexes[0].cpu(), expected)
    assert task.req_obj is req
    assert task.move_token_num == 64
    assert req.cpu_cache_task_status == InferReq._CpuCacheTaskStatus.RUNNING
