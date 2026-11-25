"""验证 CPU 完成事件与 PD 接收生命周期的交接，不依赖真实传输进程。"""

import queue
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl import PDDecodeNode
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl_for_dp import PDDPForDecodeNode
from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import (
    LoadTask,
    LoadTaskHelper,
    MultiLevelKvCacheModule,
)

Status = InferReq._CpuCacheTaskStatus


@pytest.fixture(params=[PDDecodeNode, PDDPForDecodeNode], ids=["single_dp", "multi_dp"])
def decode_request(request, monkeypatch):
    backend = request.param.__new__(request.param)
    backend.args = SimpleNamespace(enable_cpu_cache=True, page_size=4, pd_kv_page_size=64)
    backend.is_master_in_dp = True
    backend.dp_rank_in_node = 0
    backend.dp_size_in_node = 1
    backend.dp_world_size = 1
    backend.node_world_size = 1
    backend.info_queue = queue.Queue()
    req = SimpleNamespace(
        req_id=17,
        req_idx=0,
        cur_kv_len=128,
        hold_kv_len=128,
        cur_output_len=0,
        pd_task_num=0,
        pd_task_success_num=0,
        pd_task_failed_num=0,
        pd_trans_device_id=-1,
        infer_aborted=False,
        filter_mark=False,
        finish_status=FinishStatus(),
        cpu_cache_load_task_status=Status.FINISHED,
        cpu_cache_load_finished_callback=None,
        cpu_load_ready=False,
        cpu_cache_offload_task_status=Status.NOT_STARTED,
        shared_kv_node=SimpleNamespace(node_prefix_total_len=64),
        shm_req=SimpleNamespace(input_len=257),
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(pd_prefill_first=False), pd_master_node_id=1),
    )
    req.kv_cache_alloc_need = lambda target: max(0, (target + 3) // 4 * 4 - req.hold_kv_len)

    def allocate(count):
        start = req.hold_kv_len
        req.hold_kv_len += count
        return torch.arange(start, start + count, dtype=torch.int32)

    req.alloc_kv_mem = allocate
    monkeypatch.setattr(g_infer_context, "requests_mapping", {req.req_id: req})
    monkeypatch.setattr(g_infer_context, "get_can_alloc_token_num", lambda: 4096)
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(g_infer_context, "add_reqs", lambda reqs, init_prefix_cache: [req])
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.backend = backend
    module.load_tasks = deque()
    module.cpu_events = []
    module.cpu_cache_client = SimpleNamespace(
        lock=SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
        deref_pages=lambda pages: module.cpu_events.append(("deref", pages)),
    )

    def load(reqs):
        assert reqs == [req]
        req.cpu_cache_load_task_status = Status.RUNNING
        module.load_tasks.append(LoadTask(req, [5], SimpleNamespace(query=lambda: req.cpu_load_ready)))

    module.load_cpu_cache_to_reqs = load
    backend.multi_level_cache_module = module
    return backend, req


def test_init_waits_for_load_then_submits_receive_once(decode_request, monkeypatch):
    backend, req = decode_request
    backend.dp_size_in_node = 2
    added = []

    def add_reqs(reqs, init_prefix_cache):
        assert init_prefix_cache
        added.extend(reqs)
        return [req]

    monkeypatch.setattr(g_infer_context, "add_reqs", add_reqs)

    assert backend._init_reqs([(17, None, None, 0), (18, None, None, 1)]) == [17]
    assert len(added) == 1
    assert backend.info_queue.empty()
    assert req.cpu_cache_load_finished_callback is not None
    backend.multi_level_cache_module.update_cpu_cache_load_task_states()
    assert backend._filter_not_ready_reqs([17]) == []
    assert backend.info_queue.empty()

    req.cpu_load_ready = True
    backend.multi_level_cache_module.update_cpu_cache_load_task_states()
    assert backend._filter_not_ready_reqs([17]) == []
    group = backend.info_queue.get_nowait()
    assert req.cpu_cache_load_finished_callback is None
    assert backend.multi_level_cache_module.cpu_events == [("deref", [5])]
    assert [task.page_kind for task in group.task_list] == ["kv", "kv", "kv", "att_state"]
    assert group.task_list[0].start_kv_index == 128
    assert group.task_list[-1].start_kv_index == 257
    assert req.cur_kv_len == 257
    assert req.hold_kv_len == 260
    for _ in range(3):
        backend.multi_level_cache_module.update_cpu_cache_load_task_states()
        assert backend._filter_not_ready_reqs([17]) == []
    assert backend.info_queue.empty()
    req.pd_task_success_num = req.pd_task_num
    assert backend._filter_not_ready_reqs([17]) == [req]


@pytest.mark.parametrize("terminal", ["abort", "finished", "filtered"])
def test_terminal_request_waits_for_io_without_submitting_pd(decode_request, terminal):
    backend, req = decode_request
    backend._init_reqs([(17, None, None, 0)])
    if terminal == "abort":
        req.infer_aborted = True
    elif terminal == "finished":
        req.finish_status.set_status(FinishStatus.FINISHED_STOP)
    else:
        req.filter_mark = True
    assert backend._filter_not_ready_reqs([17]) == []
    req.cpu_load_ready = True
    backend.multi_level_cache_module.update_cpu_cache_load_task_states()
    req.cpu_cache_offload_task_status = Status.RUNNING
    assert backend._filter_not_ready_reqs([17]) == []
    req.cpu_cache_offload_task_status = Status.FINISHED
    assert backend._filter_not_ready_reqs([17]) == [req]
    assert backend.info_queue.empty()
    assert req.cpu_cache_load_finished_callback is None


def test_capacity_is_rechecked_before_any_task_is_published(decode_request, monkeypatch):
    backend, req = decode_request
    backend._init_reqs([(17, None, None, 0)])
    monkeypatch.setattr(g_infer_context, "get_can_alloc_token_num", lambda: 128)
    req.alloc_kv_mem = lambda count: pytest.fail("must check capacity before allocating")
    req.cpu_load_ready = True
    backend.multi_level_cache_module.update_cpu_cache_load_task_states()
    assert backend._filter_not_ready_reqs([17]) == [req]
    assert req.finished_by_pd_decode_capacity and req.filter_mark
    assert req.cur_kv_len == req.hold_kv_len == 128
    assert req.pd_task_num == 0 and req.cpu_cache_load_finished_callback is None
    assert backend._filter_not_ready_reqs([17]) == [req]
    assert backend.info_queue.empty()


def test_dense_cpu_full_match_waits_for_h2d_before_zero_length_handshake(decode_request, monkeypatch):
    backend, req = decode_request
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", False)
    req.cur_kv_len = req.hold_kv_len = 256
    backend._init_reqs([(17, None, None, 0)])
    assert backend._filter_not_ready_reqs([17]) == []
    assert backend.info_queue.empty()
    req.cpu_load_ready = True
    backend.multi_level_cache_module.update_cpu_cache_load_task_states()
    assert backend._filter_not_ready_reqs([17]) == []
    tasks = backend.info_queue.get_nowait().task_list
    assert [(task.page_kind, task.start_kv_index, task.end_kv_index) for task in tasks] == [("kv", 256, 256)]
    assert req.cur_kv_len == req.hold_kv_len == 256
    req.pd_task_success_num = 1
    assert backend._filter_not_ready_reqs([17]) == [req]


@pytest.mark.parametrize("prefill_first", [False, True])
def test_hybrid_gpu_full_match_and_prefill_first_have_distinct_handshakes(decode_request, prefill_first):
    backend, req = decode_request
    req.sampling_param.shm_param.pd_prefill_first = prefill_first
    req.cur_kv_len = req.hold_kv_len = 256  # radix 已恢复 KV + S(256)
    backend._init_pd_receive_tasks(req)
    tasks = backend.info_queue.get_nowait().task_list
    assert [(task.page_kind, task.start_kv_index, task.end_kv_index) for task in tasks] == (
        [("kv", 256, 257), ("att_state", 257, 257)] if prefill_first else [("kv", 256, 256)]
    )


def test_transfer_failure_preserves_physical_tail_but_invalidates_logical_kv(decode_request):
    backend, req = decode_request
    req.shm_req.mark_simulated_finished = lambda *args, **kwargs: None
    backend._init_pd_receive_tasks(req)
    held = req.hold_kv_len
    req.pd_task_failed_num = req.pd_task_num
    assert backend._filter_not_ready_reqs([17]) == [req]
    assert req.finish_status.is_finished()
    assert req.cur_kv_len == req.shm_req.shm_cur_kv_len == 64
    assert req.hold_kv_len == held


@pytest.mark.parametrize("prefill_first", [False, True])
def test_single_token_prompt_has_an_initial_state_and_a_complete_handshake(decode_request, prefill_first):
    backend, req = decode_request
    req.shm_req.input_len = 1
    req.cur_kv_len = req.hold_kv_len = 0
    req.sampling_param.shm_param.pd_prefill_first = prefill_first
    backend._init_pd_receive_tasks(req)
    tasks = backend.info_queue.get_nowait().task_list
    assert [(task.page_kind, task.start_kv_index, task.end_kv_index) for task in tasks] == (
        [("kv", 0, 1), ("att_state", 1, 1)] if prefill_first else [("kv", 0, 0)]
    )
    assert req.pd_task_num == len(tasks)


@pytest.mark.parametrize("cpu_cache", [False, True], ids=["cache_disabled", "cache_miss"])
def test_no_load_continues_pd_initialization_immediately(decode_request, cpu_cache):
    backend, req = decode_request
    backend.args.enable_cpu_cache = cpu_cache
    module = backend.multi_level_cache_module
    module.load_cpu_cache_to_reqs = lambda reqs: LoadTaskHelper(module).release_uncopied_pages([(req, [])])
    assert backend._init_reqs([(17, None, None, 0)]) == [17]
    assert not backend.info_queue.empty()
    assert req.cpu_cache_load_finished_callback is None
    assert not module.load_tasks
