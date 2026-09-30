import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process
from lightllm.server.router.model_infer.mode_backend.pd.trans_process_obj import KVTransProcess


@pytest.mark.parametrize("status, expected", [("module_ready", True), ("init_failed", False)])
def test_transfer_process_requires_module_ready(status, expected):
    worker = KVTransProcess(process=Mock(), task_out_queue=queue.Queue(), device_id=0)
    worker.task_out_queue.put(status)
    assert worker.wait_until_ready() is expected


def test_transfer_process_waits_through_slow_initialization():
    output = Mock()
    output.get.side_effect = [queue.Empty(), queue.Empty(), "module_ready"]
    worker = KVTransProcess(process=Mock(), task_out_queue=output, device_id=0)
    worker.process.is_alive.return_value = True
    assert worker.wait_until_ready()
    assert output.get.call_count == 3


def test_transfer_process_detects_exit_before_ready():
    worker = KVTransProcess(process=Mock(), task_out_queue=Mock(), device_id=0)
    worker.task_out_queue.get.side_effect = queue.Empty()
    worker.process.is_alive.return_value = False
    assert not worker.wait_until_ready()
    assert worker.task_out_queue.get.call_count == 1


def test_transfer_process_initialization_has_bounded_wait():
    worker = KVTransProcess(process=Mock(), task_out_queue=Mock(), device_id=0)
    worker.task_out_queue.get.side_effect = queue.Empty()
    worker.process.is_alive.return_value = True
    assert not worker.wait_until_ready()
    assert worker.task_out_queue.get.call_count == 600


def make_task(request_id, page, started=None):
    task = SimpleNamespace(request_id=request_id, start_trans_time=started, time_out_secs=10)
    task.get_key = lambda: f"{request_id}:{page}"
    task.time_out = Mock(return_value=False)
    return task


def test_waiting_pages_timeout_from_request_progress_not_creation(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: now[0])
    module = decode_trans_process._DecodeTransModule.__new__(decode_trans_process._DecodeTransModule)
    module.waiting_dict_lock = threading.Lock()
    first, second, unrelated = make_task(1, 0), make_task(1, 1), make_task(2, 0)
    module.waiting_dict = {task.get_key(): task for task in (first, second, unrelated)}
    module.request_last_progress_time = {}
    module.failed_queue = queue.Queue()

    # No transfer progress yet: the PD master owns the prefill-stage deadline.
    module._check_tasks_time_out()
    assert len(module.waiting_dict) == 3
    assert module._pop_waiting_task_for_notify(first) is first
    assert module.request_last_progress_time == {1: 100.0}

    now[0] = 109.0
    module._check_tasks_time_out()
    assert second.get_key() in module.waiting_dict
    now[0] = 111.0
    module._check_tasks_time_out()
    assert module.failed_queue.get_nowait() is second
    assert unrelated.get_key() in module.waiting_dict
    assert not module.request_last_progress_time
    assert module._pop_waiting_task_for_notify(first) is None


def test_started_transfer_keeps_per_page_timeout(monkeypatch):
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: 100.0)
    module = decode_trans_process._DecodeTransModule.__new__(decode_trans_process._DecodeTransModule)
    task = make_task(1, 0, started=1.0)
    task.time_out.return_value = True
    module.waiting_dict = {task.get_key(): task}
    module.waiting_dict_lock = threading.Lock()
    module.request_last_progress_time = {1: 99.0}
    module.failed_queue = queue.Queue()
    module._check_tasks_time_out()
    task.time_out.assert_called_once_with()
    assert module.failed_queue.get_nowait() is task
    assert not module.waiting_dict and not module.request_last_progress_time
