import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lightllm.server.pd_io_struct import PDAbortReq
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl import PDDecodeNode
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl import prefill_impl, prefill_trans_process
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
    PDChunkedPrefillForPrefillNode,
)
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_kv_move_manager import (
    PrefillKVMoveManager,
)
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_trans_process import (
    _PrefillTransModule,
)


class _StopLoop(Exception):
    pass


class _SequenceQueue:
    def __init__(self, *values):
        self.values = list(values)

    def get(self):
        if self.values:
            return self.values.pop(0)
        raise _StopLoop()


def test_prefill_infer_forwards_abort_to_kv_move_manager(monkeypatch):
    request_id = 123
    req = SimpleNamespace(
        req_id=request_id,
        infer_aborted=True,
        pd_task_num=2,
        pd_task_failed_num=0,
        pd_task_success_num=0,
        pd_trans_device_id=1,
        cur_kv_len=4,
        shm_req=SimpleNamespace(input_len=8),
    )
    monkeypatch.setattr(prefill_impl.g_infer_context, "requests_mapping", {request_id: req})

    backend = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    backend.is_master_in_dp = True
    backend.info_queue = queue.Queue()

    ready_reqs = backend._filter_not_ready_reqs([request_id])

    assert ready_reqs == []
    abort_req = backend.info_queue.get_nowait()
    assert abort_req == PDAbortReq(request_id=request_id, device_id=1)
    assert req.pd_abort_req_send_count == 1


def test_prefill_infer_sends_abort_at_most_six_times(monkeypatch):
    request_id = 123
    req = SimpleNamespace(
        req_id=request_id,
        infer_aborted=True,
        pd_task_num=2,
        pd_task_failed_num=0,
        pd_task_success_num=0,
        pd_trans_device_id=1,
        cur_kv_len=4,
        shm_req=SimpleNamespace(input_len=8),
    )
    monkeypatch.setattr(prefill_impl.g_infer_context, "requests_mapping", {request_id: req})

    backend = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    backend.is_master_in_dp = True
    backend.info_queue = queue.Queue()

    for _ in range(8):
        assert backend._filter_not_ready_reqs([request_id]) == []

    abort_reqs = [backend.info_queue.get_nowait() for _ in range(6)]
    assert all(abort_req == PDAbortReq(request_id=request_id, device_id=1) for abort_req in abort_reqs)
    assert backend.info_queue.empty()
    assert req.pd_abort_req_send_count == 6


def test_decode_infer_sends_abort_at_most_six_times(monkeypatch):
    request_id = 123
    req = SimpleNamespace(
        req_id=request_id,
        infer_aborted=True,
        pd_task_num=2,
        pd_task_failed_num=0,
        pd_task_success_num=0,
        pd_trans_device_id=1,
    )
    monkeypatch.setattr(decode_impl.g_infer_context, "requests_mapping", {request_id: req})

    backend = PDDecodeNode.__new__(PDDecodeNode)
    backend.is_master_in_dp = True
    backend.info_queue = queue.Queue()

    for _ in range(8):
        assert backend._filter_not_ready_reqs([request_id]) == []

    abort_reqs = [backend.info_queue.get_nowait() for _ in range(6)]
    assert all(abort_req == PDAbortReq(request_id=request_id, device_id=1) for abort_req in abort_reqs)
    assert backend.info_queue.empty()
    assert req.pd_abort_req_send_count == 6


def test_prefill_kv_move_manager_routes_abort_to_target_device():
    abort_req = PDAbortReq(request_id=123, device_id=1)
    target_queue = queue.Queue()
    manager = PrefillKVMoveManager.__new__(PrefillKVMoveManager)
    manager.info_queue = _SequenceQueue(abort_req)
    manager.kv_trans_processes = [
        SimpleNamespace(task_in_queue=queue.Queue()),
        SimpleNamespace(task_in_queue=target_queue),
    ]

    with pytest.raises(_StopLoop):
        manager.task_dispatcher_loop()

    assert target_queue.get_nowait() is abort_req
    assert manager.kv_trans_processes[0].task_in_queue.empty()


def test_prefill_trans_process_handles_abort_without_allocating_page(monkeypatch):
    abort_req = PDAbortReq(request_id=123, device_id=0)
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.device_id = 0
    module.task_in_queue = _SequenceQueue(abort_req)
    module.page_index_queue = MagicMock()
    module._abort = MagicMock()
    monkeypatch.setattr(prefill_trans_process.torch.cuda, "set_device", lambda _device_id: None)

    with pytest.raises(_StopLoop):
        module.recv_task_loop()

    module._abort.assert_called_once_with(request_id=123)
    module.page_index_queue.get.assert_not_called()


def test_prefill_abort_only_fails_tasks_not_submitted_to_transporter():
    pending_task = SimpleNamespace(request_id=123, xfer_handle=None, error_info=None)
    active_task = SimpleNamespace(request_id=123, xfer_handle=object(), error_info=None)
    unrelated_task = SimpleNamespace(request_id=456, xfer_handle=None, error_info=None)
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {
        "pending": pending_task,
        "active": active_task,
        "unrelated": unrelated_task,
    }
    module.failed_queue = queue.Queue()

    module._abort(request_id=123)

    assert module.failed_queue.get_nowait() is pending_task
    assert pending_task.error_info == "aborted req"
    assert module.waiting_dict == {
        "active": active_task,
        "unrelated": unrelated_task,
    }


@pytest.mark.parametrize("telemetry", [False, True])
@pytest.mark.parametrize("notify_fails", [False, True])
def test_completed_transfer_reaches_recycling_queue_with_compact_logging(monkeypatch, telemetry, notify_fails):
    task = SimpleNamespace(
        xfer_handle=object(),
        request_id=123,
        error_info=None,
        get_key=lambda: "transfer-123",
        to_str=lambda: "request_id=123",
    )
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {task.get_key(): task}
    module.success_queue, module.failed_queue = queue.Queue(), queue.Queue()
    module.transporter = MagicMock()
    module.transporter.check_task_status.return_value = "DONE"
    module.transporter.capture_telemetry = telemetry
    module.transporter.nixl_agent.get_xfer_telemetry.return_value = SimpleNamespace(
        totalBytes=4096,
        xferDuration=20,
        postDuration=2,
    )
    logger = MagicMock()
    logger.isEnabledFor.return_value = True
    monkeypatch.setattr(prefill_trans_process, "logger", logger)
    if notify_fails:
        module.transporter.send_write_done_task_to_decode_node.side_effect = ConnectionError("peer gone")
    module._update_task_status_once()
    assert not module.waiting_dict
    if notify_fails:
        assert module.failed_queue.get_nowait() is task
        assert module.success_queue.empty()
        logger.exception.assert_called_once()
    else:
        assert module.success_queue.get_nowait() is task
        assert module.failed_queue.empty()
    if telemetry:
        module.transporter.nixl_agent.get_xfer_telemetry.assert_called_once_with(task.xfer_handle)
        assert logger.debug.call_args.kwargs["extra"]["bytes_total"] == 4096
    else:
        module.transporter.nixl_agent.get_xfer_telemetry.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_optional_cache_compaction_debug_is_rate_limited_and_tp_leader_only(monkeypatch, rank):
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.utils import log_utils

    now = [0.0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: now[0])
    backend = ModeBackend.__new__(ModeBackend)
    backend.rank_in_dp = rank
    backend.logger = MagicMock()
    backend.logger.isEnabledFor.return_value = True
    backend._radix_tree_merge_counter = 0
    backend._enable_radix_tree_timer_merge = True
    backend._radix_tree_merge_update_delta = 1
    backend.radix_cache = MagicMock()
    for _ in range(2000):
        backend._timer_merge_radix_tree()
    assert backend.radix_cache.merge_unreferenced_nodes.call_count == 2000
    if rank == 0:
        assert backend.logger.debug.call_count == 1
        now[0] = 10
        backend._debug_compactions.flush()
        assert backend.logger.debug.call_count == 2
        assert sum(call.kwargs["extra"]["count"] for call in backend.logger.debug.call_args_list) == 2000
    else:
        backend.logger.debug.assert_not_called()
