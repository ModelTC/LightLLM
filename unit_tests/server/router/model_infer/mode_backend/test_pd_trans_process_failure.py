import queue
import threading

import pytest

from lightllm.server.pd_io_struct import PDAbortReq, PDChunckedTransTask, PDChunckedTransTaskGroup
from lightllm.server.router.model_infer.mode_backend.pd import base_kv_move_manager as manager_module


class FakeTransProcess:
    def __init__(self, healthy, init_result=True, ready=True):
        self.healthy = healthy
        self.init_result = init_result
        self.ready = ready
        self.killed = False
        self.process = None
        self.device_id = None
        self.task_in_queue = queue.Queue()
        self.task_out_queue = queue.Queue()

    def is_trans_process_health(self):
        return self.healthy

    def killself(self):
        self.killed = True

    def init_all(self, **kwargs):
        self.init_args = kwargs
        self.device_id = kwargs["device_id"]
        return self.init_result

    def wait_until_ready(self):
        return self.ready


def test_unhealthy_transfer_worker_exits_manager_without_signaling_router(monkeypatch):
    manager = manager_module.BaseKVMoveManager.__new__(manager_module.BaseKVMoveManager)
    manager.node_world_size = 2
    manager.kv_trans_processes = [FakeTransProcess(False), FakeTransProcess(True)]
    manager._trans_process_locks = [threading.Lock(), threading.Lock()]
    manager._pending_trans_tasks_by_device = [{}, {}]
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
    manager._pending_trans_tasks_by_device = [{}]
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


def _make_trans_task(
    request_id: int, start_kv_index: int = 0, end_kv_index: int = 1, page_kind: str = "kv"
) -> PDChunckedTransTask:
    return PDChunckedTransTask(
        request_id=request_id,
        start_kv_index=start_kv_index,
        end_kv_index=end_kv_index,
        time_out_secs=10,
        pd_master_node_id=0,
        prefill_dp_index=0,
        decode_dp_index=0,
        src_device_id=0,
        dst_device_id=0,
        mem_indexes=[0] * (end_kv_index - start_kv_index),
        prefill_agent_name=None,
        prefill_agent_metadata=None,
        prefill_num_pages=None,
        prefill_page_reg_desc=None,
        decode_agent_name=None,
        decode_agent_metadata=None,
        decode_num_pages=None,
        decode_page_reg_desc=None,
        first_gen_token_id=None,
        first_gen_token_logprob=None,
        page_kind=page_kind,
    )


class NoReadQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get(self):
        raise AssertionError("restart must not read the stopped worker's input queue")


def test_restarting_worker_fails_pending_tasks_without_reading_input_queue(monkeypatch):
    manager = manager_module.BaseKVMoveManager.__new__(manager_module.BaseKVMoveManager)
    old_process = FakeTransProcess(True)
    old_process.device_id = 0
    old_process.task_in_queue = NoReadQueue()
    new_process = FakeTransProcess(True)
    tasks = [
        _make_trans_task(1),
        _make_trans_task(2),
        _make_trans_task(3),
        _make_trans_task(4, start_kv_index=5, end_kv_index=5),
        _make_trans_task(4, start_kv_index=5, end_kv_index=5, page_kind="att_state"),
    ]
    manager.node_world_size = 1
    manager.kv_trans_processes = [old_process]
    manager._trans_process_locks = [threading.Lock()]
    manager._pending_trans_tasks_by_device = [{}]
    manager._trans_process_restart_counts = [0]
    manager._trans_process_last_restart_at = [None]
    manager.ret_obj_queue = queue.Queue()
    manager.args = object()
    manager.start_trans_process_func = object()
    manager.up_status_in_queue = None
    monkeypatch.setattr(manager_module, "KVTransProcess", lambda: new_process)

    manager.dispatch_trans_task(0, tasks[0])
    manager.dispatch_trans_task(0, PDChunckedTransTaskGroup(task_list=tasks[1:]))
    manager.dispatch_trans_task(0, PDAbortReq(request_id=5, device_id=0))
    assert len(old_process.task_in_queue.items) == 3

    old_process.healthy = False
    old_process.task_out_queue.put(tasks[0].createRetObj())
    manager._restart_trans_process(0)
    retry_task = _make_trans_task(1)
    manager.dispatch_trans_task(0, retry_task)
    manager.task_ret_handle_loop(old_process)

    returns = [manager.ret_obj_queue.get_nowait() for _ in tasks]
    assert len(returns) == len(tasks)
    assert sorted(ret.request_id for ret in returns) == [1, 2, 3, 4, 4]
    assert all(ret.has_error for ret in returns)
    assert all("restarted before reporting task result" in ret.error_info for ret in returns)
    assert old_process.task_out_queue.empty()
    assert len(manager._pending_trans_tasks_by_device[0][retry_task.createRetObj().get_key()]) == 1

    new_process.task_out_queue.put(retry_task.createRetObj())
    new_process.task_out_queue.put(None)
    retry_return = manager.ret_obj_queue.get(timeout=1)
    assert retry_return.request_id == retry_task.request_id
    assert not retry_return.has_error
    assert not manager._pending_trans_tasks_by_device[0]


def test_worker_return_clears_pending_task_before_upload():
    manager = manager_module.BaseKVMoveManager.__new__(manager_module.BaseKVMoveManager)
    trans_process = FakeTransProcess(True)
    trans_process.device_id = 0
    manager.kv_trans_processes = [trans_process]
    manager._trans_process_locks = [threading.Lock()]
    manager._pending_trans_tasks_by_device = [{}]
    manager.ret_obj_queue = queue.Queue()
    task = _make_trans_task(1)
    manager.dispatch_trans_task(0, task)
    trans_process.task_out_queue.put(task.createRetObj())
    trans_process.task_out_queue.put(None)
    manager.task_ret_handle_loop(trans_process)

    ret = manager.ret_obj_queue.get_nowait()
    assert ret.request_id == 1
    assert not ret.has_error
    assert not manager._pending_trans_tasks_by_device[0]


def test_dispatch_waits_for_a_healthy_replacement():
    checked_health = threading.Event()

    class UnhealthyTransProcess(FakeTransProcess):
        def is_trans_process_health(self):
            checked_health.set()
            return False

    manager = manager_module.BaseKVMoveManager.__new__(manager_module.BaseKVMoveManager)
    task = _make_trans_task(1)
    old_process = UnhealthyTransProcess(False)
    new_process = FakeTransProcess(True)
    manager.kv_trans_processes = [old_process]
    manager._trans_process_locks = [threading.Lock()]
    manager._pending_trans_tasks_by_device = [{}]
    dispatcher = threading.Thread(target=manager.dispatch_trans_task, args=(0, task))
    dispatcher.start()

    assert checked_health.wait(timeout=1)
    with manager._trans_process_locks[0]:
        manager.kv_trans_processes[0] = new_process
    dispatcher.join(timeout=1)

    assert not dispatcher.is_alive()
    assert new_process.task_in_queue.get_nowait() is task
