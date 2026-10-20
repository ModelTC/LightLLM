import queue
from unittest.mock import Mock

import pytest

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
