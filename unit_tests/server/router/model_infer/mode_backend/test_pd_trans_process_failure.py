import queue
import threading

import pytest

from lightllm.server.router.model_infer.mode_backend.pd import base_kv_move_manager as manager_module


class FakeTransProcess:
    def __init__(self, healthy, init_result=True, ready=True):
        self.healthy = healthy
        self.init_result = init_result
        self.ready = ready
        self.killed = False
        self.process = None
        self.task_in_queue = queue.Queue()
        self.task_out_queue = queue.Queue()

    def is_trans_process_health(self):
        return self.healthy

    def killself(self):
        self.killed = True

    def init_all(self, **kwargs):
        self.init_args = kwargs
        return self.init_result

    def wait_until_ready(self):
        return self.ready


def test_unhealthy_transfer_worker_exits_manager_without_signaling_router(monkeypatch):
    manager = manager_module.BaseKVMoveManager.__new__(manager_module.BaseKVMoveManager)
    manager.node_world_size = 2
    manager.kv_trans_processes = [FakeTransProcess(False), FakeTransProcess(True)]
    manager._trans_process_locks = [threading.Lock(), threading.Lock()]
    manager._trans_process_restart_counts = [0, 0]
    manager._trans_process_last_restart_at = [None, None]
    manager.args = object()
    manager.start_trans_process_func = object()
    manager.up_status_in_queue = None

    def failed_worker():
        return FakeTransProcess(False, init_result=False)

    monkeypatch.setattr(manager_module, "KVTransProcess", failed_worker)

    def exit_manager(code):
        raise SystemExit(code)

    monkeypatch.setattr(manager_module.os, "kill", lambda *args: pytest.fail("manager signaled a process"))
    monkeypatch.setattr(manager_module.os, "_exit", exit_manager)

    with pytest.raises(SystemExit) as exc_info:
        manager.check_trans_process_loop()

    assert exc_info.value.code == 1
    assert all(process.killed for process in manager.kv_trans_processes)


def test_unhealthy_transfer_worker_restarts_and_reopens_return_handler(monkeypatch):
    manager = manager_module.BaseKVMoveManager.__new__(manager_module.BaseKVMoveManager)
    old_process = FakeTransProcess(False)
    new_process = FakeTransProcess(True)
    new_process.task_out_queue.put(None)
    manager.node_world_size = 1
    manager.kv_trans_processes = [old_process]
    manager._trans_process_locks = [threading.Lock()]
    manager._trans_process_restart_counts = [0]
    manager._trans_process_last_restart_at = [None]
    manager.ret_obj_queue = queue.Queue()
    manager.args = object()
    manager.start_trans_process_func = object()
    manager.up_status_in_queue = None
    monkeypatch.setattr(manager_module, "KVTransProcess", lambda: new_process)

    manager._restart_trans_process(0)

    assert manager.kv_trans_processes == [new_process]
    assert new_process.init_args["device_id"] == 0
    assert new_process.init_args["manager"] is manager
    assert old_process.task_out_queue.get_nowait() is None
    assert manager._trans_process_last_restart_at[0] is not None


def test_dispatch_waits_for_a_healthy_replacement():
    checked_health = threading.Event()

    class UnhealthyTransProcess(FakeTransProcess):
        def is_trans_process_health(self):
            checked_health.set()
            return False

    manager = manager_module.BaseKVMoveManager.__new__(manager_module.BaseKVMoveManager)
    task = object()
    old_process = UnhealthyTransProcess(False)
    new_process = FakeTransProcess(True)
    manager.kv_trans_processes = [old_process]
    manager._trans_process_locks = [threading.Lock()]
    dispatcher = threading.Thread(target=manager.dispatch_trans_task, args=(0, task))
    dispatcher.start()

    assert checked_health.wait(timeout=1)
    with manager._trans_process_locks[0]:
        manager.kv_trans_processes[0] = new_process
    dispatcher.join(timeout=1)

    assert not dispatcher.is_alive()
    assert new_process.task_in_queue.get_nowait() is task
