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
