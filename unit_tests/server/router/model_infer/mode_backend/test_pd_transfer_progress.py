import pickle
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lightllm.server.pd_io_struct import PDChunckedTransTask, PDChunckedTransTaskGroup
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process


class _StopLoop(Exception):
    pass


class _SequenceQueue:
    def __init__(self, *values):
        self.values = list(values)

    def get(self):
        if self.values:
            return self.values.pop(0)
        raise _StopLoop()


def make_task(request_id, page, empty=False):
    return PDChunckedTransTask(
        request_id=request_id,
        start_kv_index=page,
        end_kv_index=page if empty else page + 1,
        time_out_secs=10,
        pd_master_node_id=1,
        prefill_dp_index=0,
        decode_dp_index=0,
        src_device_id=0,
        dst_device_id=0,
        mem_indexes=[] if empty else [page],
        prefill_agent_name="prefill",
        prefill_agent_metadata=b"prefill metadata",
        prefill_num_pages=1,
        prefill_page_reg_desc=b"prefill page",
        decode_agent_name=None,
        decode_agent_metadata=None,
        decode_num_pages=None,
        decode_page_reg_desc=None,
        first_gen_token_id=42,
        first_gen_token_logprob=-0.5,
    )


def make_module(*tasks):
    module = decode_trans_process._DecodeTransModule.__new__(decode_trans_process._DecodeTransModule)
    module.device_id = 0
    module.use_write_quiescence = False
    module.args = SimpleNamespace(pd_node_id=2, pd_detail_log=False)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {task.get_key(): task for task in tasks}
    module.transporter = SimpleNamespace(
        agent_name="decode",
        agent_metadata=b"decode metadata",
        num_pages=1,
        local_page_mem_desc=b"decode page",
        send_error_info_to_prefill_node=Mock(),
    )
    for name in (
        "request_page_task_queue",
        "ready_page_task_queue",
        "success_queue",
        "failed_queue",
        "up_status_in_queue",
    ):
        setattr(module, name, queue.Queue())
    return module


def test_dispatch_shares_timeout_mark_with_all_pages_in_each_group(monkeypatch):
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: 100.0)
    first, second, empty, unrelated = make_task(1, 0), make_task(1, 1), make_task(1, 2, empty=True), make_task(2, 0)
    module = make_module()
    module.recv_task_group_queue = _SequenceQueue(
        PDChunckedTransTaskGroup([first, second, empty]), PDChunckedTransTaskGroup([unrelated])
    )

    with pytest.raises(_StopLoop):
        module.dispatch_task_loop()

    assert first.shared_time_out_mark is second.shared_time_out_mark is empty.shared_time_out_mark
    assert unrelated.shared_time_out_mark is not first.shared_time_out_mark
    assert first.shared_time_out_mark.time_mark is None
    assert module.waiting_dict == {task.get_key(): task for task in (first, second, unrelated)}
    assert module.success_queue.get_nowait() == (None, None, empty)
    assert empty.start_trans_time == 100.0
    restored = pickle.loads(pickle.dumps(PDChunckedTransTaskGroup([first, second, empty]))).task_list
    assert restored[0].shared_time_out_mark is restored[1].shared_time_out_mark is restored[2].shared_time_out_mark


def test_first_notify_wait_does_not_charge_prefill_queueing(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: now[0])
    first, second = make_task(1, 0), make_task(1, 1)
    module = make_module()
    module.recv_task_group_queue = _SequenceQueue(PDChunckedTransTaskGroup([first, second]))

    with pytest.raises(_StopLoop):
        module.dispatch_task_loop()

    now[0] = 1000.0
    module._check_tasks_time_out()
    assert module.failed_queue.empty()
    assert len(module.waiting_dict) == 2
    assert first.shared_time_out_mark.time_mark is None


def test_waiting_pages_timeout_from_shared_progress(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: now[0])
    first, second, unrelated = make_task(1, 0), make_task(1, 1), make_task(2, 0)
    first.shared_time_out_mark = second.shared_time_out_mark = decode_trans_process._SharedTimeOutMark()
    unrelated.shared_time_out_mark = decode_trans_process._SharedTimeOutMark()
    unrelated.shared_time_out_mark.update()
    for task in (first, second, unrelated):
        task.create_time = 0.0
    module = make_module(first, second, unrelated)

    now[0] = 109.0
    module._check_tasks_time_out()
    assert len(module.waiting_dict) == 3
    first.shared_time_out_mark.update()
    now[0] = 111.0
    module._check_tasks_time_out()
    assert module.failed_queue.get_nowait() is unrelated
    assert len(module.waiting_dict) == 2
    now[0] = 120.0
    module._check_tasks_time_out()
    assert [module.failed_queue.get_nowait(), module.failed_queue.get_nowait()] == [first, second]
    assert not module.waiting_dict


@pytest.mark.parametrize("stage", ["request", "done"])
def test_matched_notifications_refresh_local_siblings_only(monkeypatch, stage):
    now = [100.0]
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: now[0])
    monkeypatch.setattr(decode_trans_process.torch.cuda, "set_device", lambda _device: None)
    first, sibling, unrelated = make_task(1, 0), make_task(1, 1), make_task(2, 0)
    first.shared_time_out_mark = sibling.shared_time_out_mark = decode_trans_process._SharedTimeOutMark()
    unrelated.shared_time_out_mark = decode_trans_process._SharedTimeOutMark()
    module = make_module(first, sibling, unrelated)
    remote = make_task(1, 0)
    remote.write_stage = stage
    remote.shared_time_out_mark = decode_trans_process._SharedTimeOutMark()
    remote.shared_time_out_mark.time_mark = -100.0
    module.transporter.get_new_notifs = lambda: {"prefill": [pickle.dumps(remote)]}

    now[0] = 1009.0
    with monkeypatch.context() as patch:
        patch.setattr(module, "_check_tasks_time_out", Mock(side_effect=_StopLoop))
        with pytest.raises(_StopLoop):
            module.accept_peer_task_loop()

    assert sibling.shared_time_out_mark.time_mark == 1009.0
    assert unrelated.shared_time_out_mark.time_mark is None
    target_queue = module.request_page_task_queue if stage == "request" else module.ready_page_task_queue
    assert target_queue.get_nowait() is first

    # The repeated notify cannot keep siblings alive once its waiting task was removed.
    now[0] = 1010.0
    with monkeypatch.context() as patch:
        patch.setattr(module, "_check_tasks_time_out", Mock(side_effect=_StopLoop))
        with pytest.raises(_StopLoop):
            module.accept_peer_task_loop()
    assert sibling.shared_time_out_mark.time_mark == 1009.0
    module.transporter.send_error_info_to_prefill_node.assert_called_once()


def test_started_transfer_keeps_real_per_page_timeout(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(decode_trans_process.time, "time", lambda: now[0])
    task = make_task(1, 0)
    task.start_trans_time = 50.0
    task.shared_time_out_mark = decode_trans_process._SharedTimeOutMark()
    module = make_module(task)

    now[0] = 111.0
    module._check_tasks_time_out()
    assert module.waiting_dict
    now[0] = 116.0
    task.shared_time_out_mark.update()
    module._check_tasks_time_out()
    assert module.waiting_dict
    now[0] = 117.0
    module._check_tasks_time_out()
    assert module.failed_queue.get_nowait() is task
    assert task.error_info == "time out in accept_peer_task_loop"
    assert not module.waiting_dict
