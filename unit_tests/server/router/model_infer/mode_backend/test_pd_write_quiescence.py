import pickle
import queue
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lightllm.server.pd_io_struct import PDChunckedTransTask
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import _DecodeTransModule
from lightllm.server.router.model_infer.mode_backend.pd import nixl_kv_transporter
from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl import prefill_trans_process
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_trans_process import (
    _PrefillTransModule,
)


class _StopLoop(Exception):
    pass


class _SequenceQueue:
    def __init__(self, *items):
        self.items = iter(items)
        self.put = MagicMock()

    def get(self):
        try:
            return next(self.items)
        except StopIteration:
            raise _StopLoop()


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setattr(prefill_trans_process.torch.cuda, "set_device", lambda _: None)


def _task(**kwargs):
    values = dict(
        request_id=123,
        start_kv_index=0,
        end_kv_index=4,
        time_out_secs=10,
        pd_master_node_id=0,
        prefill_dp_index=0,
        decode_dp_index=0,
        src_device_id=0,
        dst_device_id=0,
        mem_indexes=[0, 1, 2, 3],
        prefill_agent_name="P",
        prefill_agent_metadata=None,
        prefill_num_pages=2,
        prefill_page_reg_desc=None,
        decode_agent_name="D",
        decode_agent_metadata=None,
        decode_num_pages=2,
        decode_page_reg_desc=None,
        first_gen_token_id=None,
        first_gen_token_logprob=None,
    )
    values.update(kwargs)
    task = PDChunckedTransTask(**values)
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import (
        _SharedTimeOutMark,
    )

    task.shared_time_out_mark = _SharedTimeOutMark()
    return task


def _module(cls):
    module = cls.__new__(cls)
    module.device_id = 0
    module.use_write_quiescence = True
    module.args = SimpleNamespace(pd_kv_page_num=2)
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {}
    module.quarantined_tasks = {}
    module.write_recoveries = {}
    module.write_peer_kv_queue = queue.Queue()
    module.transfer_status = MagicMock()
    module.request_last_progress_time = {}
    module.last_stop_retry_time = 0.0
    module.failed_queue = queue.Queue()
    module.success_queue = queue.Queue()
    module.page_index_queue = queue.Queue()
    module.task_out_queue = queue.Queue()
    module.transporter = SimpleNamespace(
        capture_telemetry=False,
        check_task_status=MagicMock(return_value="PROC"),
        release_xfer_handle=MagicMock(),
        send_error_info_to_decode_node=MagicMock(),
        send_error_info_to_prefill_node=MagicMock(),
        send_write_done_task_to_decode_node=MagicMock(),
    )
    return module


def _fail_one(module):
    task = module.failed_queue.get_nowait()
    module.failed_queue = _SequenceQueue(task)
    with pytest.raises(_StopLoop):
        module.fail_loop()
    module.failed_queue = queue.Queue()


def _deliver(module, task, monkeypatch):
    module.transporter.get_new_notifs = lambda: {"peer": [pickle.dumps(task)]}
    with monkeypatch.context() as patch:
        patch.setattr(module, "_check_tasks_time_out", MagicMock(side_effect=_StopLoop()))
        with pytest.raises(_StopLoop):
            if isinstance(module, _PrefillTransModule):
                module.accept_decode_write_task_loop()
            else:
                module.accept_peer_task_loop()


def test_decode_timeout_holds_page_until_matching_stop_confirmation(monkeypatch):
    module = _module(_DecodeTransModule)
    task = _task(dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    module.waiting_dict[task.get_key()] = task
    module._check_tasks_time_out()
    assert module.quarantined_tasks == {task.write_id: task}
    _fail_one(module)
    assert module.page_index_queue.empty()
    assert module.task_out_queue.qsize() == 1

    stopped = replace(task, write_stage="stopped", write_quiesced=True)
    assert not module._confirm_write_quiesced(replace(stopped, write_id="old-allocation"))
    assert not module._confirm_write_quiesced(replace(stopped, dst_page_index=0))
    assert not module._confirm_write_quiesced(replace(stopped, prefill_agent_name="other-P"))
    assert module.page_index_queue.empty()
    assert module._confirm_write_quiesced(stopped)
    assert module.page_index_queue.get_nowait() == 1
    assert not module._confirm_write_quiesced(stopped)
    assert module.page_index_queue.empty()
    assert module.task_out_queue.qsize() == 1


def test_done_between_timeout_and_failure_receipt_recycles_once_without_unpack(monkeypatch):
    module = _module(_DecodeTransModule)
    task = _task(dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    module.waiting_dict[task.get_key()] = task
    module.ready_page_task_queue = queue.Queue()
    module._check_tasks_time_out()
    done = replace(task, write_stage="done", error_info=None, write_quiesced=True)
    _deliver(module, done, monkeypatch)
    _fail_one(module)
    assert module.ready_page_task_queue.empty()
    assert module.page_index_queue.qsize() == module.task_out_queue.qsize() == 1


def test_decode_unknown_error_does_not_prove_remote_write_stopped(monkeypatch):
    module = _module(_DecodeTransModule)
    task = _task(dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    module.waiting_dict[task.get_key()] = task
    _deliver(module, replace(task, write_stage="error", error_info="state query failed"), monkeypatch)
    _fail_one(module)
    assert module.quarantined_tasks == {task.write_id: task}
    assert module.page_index_queue.empty()


def test_decode_retries_lost_stop_notification(monkeypatch):
    module = _module(_DecodeTransModule)
    task = _task(dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    module.waiting_dict[task.get_key()] = task
    module._check_tasks_time_out()
    _fail_one(module)
    module.transporter.send_error_info_to_prefill_node.reset_mock()
    monkeypatch.setattr(prefill_trans_process.time, "time", lambda: module.last_stop_retry_time + 2)
    module._check_tasks_time_out()
    module.transporter.send_error_info_to_prefill_node.assert_called_once_with(trans_task=task)
    assert module.page_index_queue.empty()


@pytest.mark.parametrize("submitting", [False, True])
def test_prefill_stop_does_not_ack_an_active_or_submitting_write(submitting):
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1")
    task.write_submitting = submitting
    task.write_submitted = not submitting
    task.xfer_handle = None if submitting else 7
    module.waiting_dict[task.get_key()] = task
    module._handle_stop_request(replace(task, write_stage="error", error_info="D timeout"))
    assert module.waiting_dict == {task.get_key(): task}
    assert task.error_info == "D timeout"
    assert module.failed_queue.empty()
    module.transporter.send_error_info_to_decode_node.assert_not_called()


def test_prefill_stop_before_submission_cancels_queued_write():
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1")
    module.waiting_dict[task.get_key()] = task
    module.write_peer_kv_queue = _SequenceQueue(task)
    module.transporter.write_blocks_paged = MagicMock()
    module._handle_stop_request(replace(task, write_stage="error", error_info="D timeout"))
    with pytest.raises(_StopLoop):
        module.write_peer_kv_loop()
    module.transporter.write_blocks_paged.assert_not_called()
    _fail_one(module)
    assert module.page_index_queue.get_nowait() == 0
    stopped = module.transporter.send_error_info_to_decode_node.call_args.kwargs["trans_task"]
    assert stopped.write_quiesced and stopped.write_id == "allocation-1"


def test_prefill_timeout_drains_write_before_reusing_source_or_confirming_stop():
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    task.write_submitted = True
    task.xfer_handle = 7
    module.waiting_dict[task.get_key()] = task
    module._update_task_status_once()
    _fail_one(module)
    assert module.page_index_queue.empty()
    module.transporter.release_xfer_handle.assert_not_called()
    assert module.task_out_queue.qsize() == 1
    assert not task.write_quiesced

    module.transporter.check_task_status.return_value = "DONE"
    module._update_task_status_once()
    assert module.page_index_queue.get_nowait() == 0
    assert module.task_out_queue.qsize() == 1
    assert task.write_quiesced
    module.transporter.release_xfer_handle.assert_called_once_with(7)
    assert not module.quarantined_tasks


@pytest.mark.parametrize("query_raises", [False, True])
def test_prefill_err_never_turns_into_a_false_done_confirmation(query_raises):
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1")
    task.write_submitted = True
    task.xfer_handle = 7
    module.waiting_dict[task.get_key()] = task
    module.transporter.check_task_status.return_value = "ERR"
    if query_raises:
        module.transporter.check_task_status.side_effect = RuntimeError("query failed")
    module._update_task_status_once()
    _fail_one(module)
    module.transporter.check_task_status.reset_mock(return_value=True, side_effect=True)
    module.transporter.check_task_status.return_value = "DONE"
    module._update_task_status_once()
    module.transporter.check_task_status.assert_not_called()
    module.transporter.release_xfer_handle.assert_not_called()
    assert module.page_index_queue.empty()
    assert not task.write_quiesced


def test_nixl_retains_handle_when_transfer_submission_raises():
    task = _task(src_page_index=0, dst_page_index=1)
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.remote_agents = {"D": object()}
    transporter._write_contexts = {}
    transporter.page_local_xfer_handles = "src"
    transporter.remote_agents["D"] = SimpleNamespace(page_xfer_handles="dst")
    transporter.nixl_agent = SimpleNamespace(make_prepped_xfer=MagicMock(return_value=7))

    def transfer(handle):
        assert task.xfer_handle == handle == 7
        assert task.write_submitted
        raise RuntimeError("partial submission")

    transporter.nixl_agent.transfer = transfer
    with pytest.raises(RuntimeError, match="partial submission"):
        transporter.write_blocks_paged(task)
    assert task.xfer_handle == 7


def test_prefill_done_before_failure_receipt_releases_resources_once():
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    task.write_submitted = True
    task.xfer_handle = 7
    module.waiting_dict[task.get_key()] = task
    module._update_task_status_once()
    module.transporter.check_task_status.return_value = "DONE"
    module._update_task_status_once()
    _fail_one(module)
    assert module.page_index_queue.qsize() == module.task_out_queue.qsize() == 1
    module.transporter.release_xfer_handle.assert_called_once_with(7)


@pytest.mark.parametrize("cls", [_PrefillTransModule, _DecodeTransModule])
@pytest.mark.parametrize("exhausted_while_waiting", [False, True])
def test_quarantined_capacity_rejects_new_transfers_without_recycling_pages(cls, exhausted_while_waiting):
    module = _module(cls)
    if exhausted_while_waiting:

        def wait_for_page(timeout):
            module.quarantined_tasks = {"a": object(), "b": object()}
            raise queue.Empty()

        module.page_index_queue = SimpleNamespace(get=wait_for_page, empty=lambda: True)
    else:
        module.quarantined_tasks = {"a": object(), "b": object()}
    task = _task()
    if cls is _PrefillTransModule:
        module.task_in_queue = _SequenceQueue(task)
        loop = module.recv_task_loop
    else:
        module.request_page_task_queue = _SequenceQueue(task)
        loop = module.request_page_loop
    with pytest.raises(_StopLoop):
        loop()
    assert module.failed_queue.get_nowait() is task
    assert "recovery required" in task.error_info
    assert module.page_index_queue.empty()


def test_prefill_submission_exception_keeps_source_page_and_handle():
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1")
    module.waiting_dict[task.get_key()] = task
    module.write_peer_kv_queue = _SequenceQueue(task)

    def submit(trans_task):
        module._handle_stop_request(replace(task, error_info="D timeout"))
        trans_task.xfer_handle = 7
        trans_task.write_submitted = True
        raise RuntimeError("partial submission")

    module.transporter.write_blocks_paged = submit
    with pytest.raises(_StopLoop):
        module.write_peer_kv_loop()
    _fail_one(module)
    assert module.quarantined_tasks == {task.get_key(): task}
    assert task.write_uncertain and not task.write_quiesced
    assert module.page_index_queue.empty()
    module.transporter.release_xfer_handle.assert_not_called()


@pytest.mark.parametrize("stage", ["done", "stopped"])
def test_late_confirmation_does_not_fail_or_free_a_new_allocation(monkeypatch, stage):
    module = _module(_DecodeTransModule)
    task = _task(dst_page_index=1, write_id="new-allocation")
    module.waiting_dict[task.get_key()] = task
    module.ready_page_task_queue = queue.Queue()
    old = replace(
        task,
        write_stage=stage,
        write_id="old-allocation",
        write_quiesced=True,
        error_info="old error" if stage == "stopped" else None,
    )
    _deliver(module, old, monkeypatch)
    assert module.waiting_dict == {task.get_key(): task}
    assert module.ready_page_task_queue.empty()
    assert module.failed_queue.empty()
    assert module.page_index_queue.empty()
    module.transporter.send_error_info_to_prefill_node.assert_not_called()


def test_normal_write_uses_existing_handshake_and_waits_for_unpack_event(monkeypatch):
    prefill, decode = _module(_PrefillTransModule), _module(_DecodeTransModule)
    src = _task(src_page_index=0, first_gen_token_id=8)
    dst = _task()
    prefill.waiting_dict[src.get_key()] = src
    decode.waiting_dict[dst.get_key()] = dst
    decode.request_page_task_queue = queue.Queue()
    decode.ready_page_task_queue = queue.Queue()
    prefill.write_peer_kv_queue = queue.Queue()

    _deliver(decode, replace(src, write_stage="request"), monkeypatch)
    decode.request_page_task_queue = _SequenceQueue(decode.request_page_task_queue.get_nowait())
    decode.page_index_queue.put(1)
    ready = []
    decode.transporter.send_write_ready_task_to_prefill_node = lambda trans_task: ready.append(
        replace(trans_task, write_stage="ready")
    )
    with pytest.raises(_StopLoop):
        decode.request_page_loop()
    assert len(ready) == 1 and ready[0].write_id is not None
    _deliver(prefill, ready[0], monkeypatch)
    prefill.write_peer_kv_queue = _SequenceQueue(prefill.write_peer_kv_queue.get_nowait())

    def submit(trans_task):
        trans_task.write_submitted = True
        trans_task.xfer_handle = 7
        return 7

    prefill.transporter.write_blocks_paged = submit
    with pytest.raises(_StopLoop):
        prefill.write_peer_kv_loop()
    done = []
    prefill.transporter.check_task_status.return_value = "DONE"
    prefill.transporter.send_write_done_task_to_decode_node = lambda task: done.append(
        replace(task, write_stage="done")
    )
    prefill._update_task_status_once()
    prefill.success_queue = _SequenceQueue(prefill.success_queue.get_nowait())
    with pytest.raises(_StopLoop):
        prefill.success_loop()
    _deliver(decode, done[0], monkeypatch)
    dst = decode.ready_page_task_queue.get_nowait()
    event = MagicMock()
    copy_start = SimpleNamespace(elapsed_time=lambda _: 1.0)
    decode.success_queue = _SequenceQueue((event, copy_start, dst))
    with pytest.raises(_StopLoop):
        decode.success_loop()
    event.synchronize.assert_called_once()
    assert prefill.page_index_queue.qsize() == decode.page_index_queue.qsize() == 1
    assert prefill.task_out_queue.qsize() == decode.task_out_queue.qsize() == 1
    assert decode.task_out_queue.get_nowait().first_gen_token_id == 8
    assert not prefill.quarantined_tasks and not decode.quarantined_tasks
    prefill.transporter.send_error_info_to_decode_node.assert_not_called()
    decode.transporter.send_error_info_to_prefill_node.assert_not_called()


def test_decode_timeout_and_prefill_drain_complete_stop_protocol(monkeypatch):
    prefill, decode = _module(_PrefillTransModule), _module(_DecodeTransModule)
    dst = _task(dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    src = replace(dst, src_page_index=0, xfer_handle=7, write_submitted=True)
    prefill.waiting_dict[src.get_key()] = src
    decode.waiting_dict[dst.get_key()] = dst
    decode._check_tasks_time_out()
    _fail_one(decode)
    assert decode.page_index_queue.empty()
    prefill._handle_stop_request(replace(dst, write_stage="error"))
    prefill.transporter.send_error_info_to_decode_node.assert_not_called()
    prefill.transporter.check_task_status.return_value = "DONE"
    prefill._update_task_status_once()
    _fail_one(prefill)
    ack = prefill.transporter.send_error_info_to_decode_node.call_args.kwargs["trans_task"]
    _deliver(decode, replace(ack, write_stage="stopped"), monkeypatch)
    assert not prefill.quarantined_tasks and not decode.quarantined_tasks
    assert prefill.page_index_queue.qsize() == decode.page_index_queue.qsize() == 1
    assert prefill.task_out_queue.qsize() == decode.task_out_queue.qsize() == 1


@pytest.mark.parametrize("quiesced", [False, True])
def test_nixl_error_notification_carries_stop_proof_and_allocation_identity(quiesced):
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.node_id, transporter.tp_idx, transporter.num_pages = 1, 0, 2
    transporter.page_reg_desc = "registered"
    transporter.remote_agents = {"D": object()}
    transporter._write_contexts = {}
    transporter.nixl_agent = SimpleNamespace(
        get_agent_metadata=lambda: b"P", get_serialized_descs=lambda _: b"pages", send_notif=MagicMock()
    )
    task = _task(write_id="allocation-1", dst_page_index=1, write_quiesced=quiesced, error_info="timeout")
    transporter.send_error_info_to_decode_node(task)
    wire = pickle.loads(transporter.nixl_agent.send_notif.call_args.kwargs["notif_msg"])
    assert wire.write_stage == ("stopped" if quiesced else "error")
    assert wire.write_quiesced == quiesced
    assert wire.write_id == task.write_id and wire.dst_page_index == task.dst_page_index


def test_lost_stop_notification_does_not_remove_active_peer_descriptors():
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.node_id, transporter.tp_idx, transporter.num_pages = 1, 0, 2
    transporter.page_reg_desc = "registered"
    transporter.remote_agents = {"D": object()}
    transporter._write_contexts = {}
    transporter.remove_remote_agent = MagicMock()
    transporter.nixl_agent = SimpleNamespace(
        get_agent_metadata=lambda: b"P",
        get_serialized_descs=lambda _: b"pages",
        send_notif=MagicMock(side_effect=RuntimeError("notification lost")),
    )
    transporter.send_error_info_to_decode_node(_task(error_info="timeout"))
    transporter.remove_remote_agent.assert_not_called()


@pytest.mark.parametrize("cls", [_PrefillTransModule, _DecodeTransModule])
@pytest.mark.parametrize("timeout_first", [False, True])
def test_nixl_notify_failure_and_timeout_have_one_cleanup_owner(cls, timeout_first):
    module = _module(cls)
    task = _task()
    task.time_out = lambda: True
    module.transporter.remove_remote_agent = MagicMock()

    def send(*args, **kwargs):
        if timeout_first:
            module._check_tasks_time_out()
        raise RuntimeError("notify failed")

    if cls is _PrefillTransModule:
        task.src_page_index = 0
        module.ready_transfer_queue = _SequenceQueue((SimpleNamespace(synchronize=lambda: None), task))
        module.transporter.send_write_request_task_to_decode_node = send
        loop = module.ready_transfer_loop
    else:
        module.page_index_queue.put(1)
        module.request_page_task_queue = _SequenceQueue(task)
        module.transporter.send_write_ready_task_to_prefill_node = send
        loop = module.request_page_loop
    with pytest.raises(_StopLoop):
        loop()
    assert module.failed_queue.qsize() == 1
    _fail_one(module)
    assert module.task_out_queue.qsize() == 1
    assert module.page_index_queue.qsize() == (1 if cls is _PrefillTransModule else 0)
    module.transporter.remove_remote_agent.assert_not_called()


def test_prefill_preparation_failure_before_submission_can_reclaim_source():
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1")
    module.waiting_dict[task.get_key()] = task
    module.write_peer_kv_queue = _SequenceQueue(task)
    module.transporter.write_blocks_paged = MagicMock(side_effect=RuntimeError("preparation failed"))
    with pytest.raises(_StopLoop):
        module.write_peer_kv_loop()
    _fail_one(module)
    assert task.write_quiesced and not task.write_submitted
    assert module.page_index_queue.get_nowait() == 0
    assert not module.quarantined_tasks


@pytest.mark.parametrize("cls", [_PrefillTransModule, _DecodeTransModule])
def test_quarantine_capacity_is_published_and_recovers_only_after_safe_page_return(cls):
    module = _module(cls)
    tasks = [
        _task(request_id=i, src_page_index=i, dst_page_index=i, write_id=f"allocation-{i}", start_trans_time=0)
        for i in range(2)
    ]
    for task in tasks:
        task.write_submitted = True
        task.xfer_handle = task.request_id
        module.waiting_dict[task.get_key()] = task
    if cls is _PrefillTransModule:
        module._update_task_status_once()
    else:
        module._check_tasks_time_out()
    assert [call.args for call in module.transfer_status.set_quarantined_pages.call_args_list] == [(0, 1), (0, 2)]
    assert module.page_index_queue.empty()

    def publish_after_page_return(device_id, count):
        assert (device_id, count) == (0, 1)
        assert module.page_index_queue.qsize() == 1

    module.transfer_status.set_quarantined_pages.side_effect = publish_after_page_return
    if cls is _PrefillTransModule:
        module.transporter.check_task_status.side_effect = ["DONE", "PROC"]
        module._update_task_status_once()
    else:
        assert module._confirm_write_quiesced(replace(tasks[0], write_quiesced=True))
    assert len(module.quarantined_tasks) == 1


def _start_recovery(module):
    recovery = module.write_peer_kv_queue.get_nowait()
    module.write_peer_kv_queue = _SequenceQueue(recovery)
    with pytest.raises(_StopLoop):
        module.write_peer_kv_loop()
    module.write_peer_kv_queue = queue.Queue()
    return recovery


def _uncertain_source(module):
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1")
    task.write_submitted = True
    task.xfer_handle = 7
    module.waiting_dict[task.get_key()] = task
    module.transporter.check_task_status.return_value = "ERR"
    module._update_task_status_once()
    _fail_one(module)
    return task


def _submit_fence(original, fence):
    assert original.src_page_index == fence.src_page_index == 0
    assert original.dst_page_index == fence.dst_page_index == 1
    assert original.write_id == fence.write_id == "allocation-1"
    fence.xfer_handle = 8
    fence.write_submitted = True
    return "PROC"


@pytest.mark.parametrize("incomplete_status", ["PROC", "UNKNOWN"])
def test_recovery_fence_requires_done_before_reclaim_and_matching_decode_stop(incomplete_status):
    module, decode = _module(_PrefillTransModule), _module(_DecodeTransModule)
    task = _uncertain_source(module)
    dst = replace(task, src_page_index=None, xfer_handle=None, write_submitted=False)
    decode.waiting_dict[dst.get_key()] = dst
    decode._fail_waiting_task_locked(dst, "failed source WRITE")
    module.transporter.start_write_fence = _submit_fence
    recovery = _start_recovery(module)
    module.transporter.check_task_status.return_value = incomplete_status
    module._update_task_status_once()
    assert module.page_index_queue.empty() and decode.page_index_queue.empty()
    assert not task.write_quiesced
    module.transporter.release_xfer_handle.assert_not_called()
    module.transporter.send_error_info_to_decode_node.reset_mock()

    recovery.next_check_time = 0
    module.transporter.check_task_status.return_value = "DONE"
    module._update_task_status_once()
    assert module.page_index_queue.get_nowait() == 0
    assert module.task_out_queue.qsize() == 1
    assert [call.args for call in module.transporter.release_xfer_handle.call_args_list] == [(8,), (7,)]
    assert task.write_quiesced
    assert not module.quarantined_tasks and not module.write_recoveries
    stopped = module.transporter.send_error_info_to_decode_node.call_args.kwargs["trans_task"]
    assert decode._confirm_write_quiesced(stopped)
    assert decode.page_index_queue.get_nowait() == 1
    assert not decode._confirm_write_quiesced(stopped)
    module._update_task_status_once()
    assert module.transporter.release_xfer_handle.call_count == 2
    assert module.task_out_queue.qsize() == 1


def test_failed_fences_remain_quarantined_and_retries_are_bounded():
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)
    handles = iter([8, 9, 10])

    def submit(original, fence):
        fence.xfer_handle = next(handles)
        fence.write_submitted = True
        return "PROC"

    module.transporter.start_write_fence = submit
    for _ in range(3):
        recovery = _start_recovery(module)
        recovery.next_check_time = 0
        module._update_task_status_once()
    assert recovery.fence.write_uncertain
    assert recovery.retired_handles == [8, 9]
    assert recovery.fence.xfer_handle == 10
    assert module.page_index_queue.empty()
    assert module.write_peer_kv_queue.empty()
    assert not task.write_quiesced
    module.transporter.release_xfer_handle.assert_not_called()
    module.transporter.check_task_status.reset_mock()
    recovery.next_check_time = 0
    module._update_task_status_once()
    module.transporter.check_task_status.assert_not_called()


def test_recovery_does_not_accept_done_after_unconfirmed_post_and_retries_with_fresh_handle():
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)

    def partial_submission(original, fence):
        _submit_fence(original, fence)
        raise RuntimeError("fence submission result lost")

    module.transporter.start_write_fence = partial_submission
    recovery = _start_recovery(module)
    module.transporter.check_task_status.return_value = "DONE"
    module._update_task_status_once()
    assert not task.write_quiesced
    assert module.page_index_queue.empty()
    module.transporter.release_xfer_handle.assert_not_called()
    assert recovery.retired_handles == [8]

    def confirmed_submission(original, fence):
        fence.xfer_handle = 9
        fence.write_submitted = True
        return "PROC"

    module.transporter.start_write_fence = confirmed_submission
    _start_recovery(module)
    recovery.next_check_time = 0
    module._update_task_status_once()
    assert task.write_quiesced
    assert module.page_index_queue.get_nowait() == 0
    assert [call.args for call in module.transporter.release_xfer_handle.call_args_list] == [(9,), (8,), (7,)]


def test_recovery_preparation_failure_retries_once_without_losing_ownership():
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)
    module.transporter.start_write_fence = MagicMock(side_effect=RuntimeError("preparation failed"))
    recovery = _start_recovery(module)
    module._update_task_status_once()
    assert module.write_peer_kv_queue.qsize() == 1
    module._queue_write_recovery_locked(task)
    module._update_task_status_once()
    assert module.write_peer_kv_queue.qsize() == 1
    assert recovery.fence.write_submitting
    assert not task.write_quiesced
    module.transporter.release_xfer_handle.assert_not_called()


def test_recovery_cleanup_failure_keeps_done_proof_and_does_not_resubmit():
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)
    module.transporter.start_write_fence = _submit_fence
    recovery = _start_recovery(module)
    module.transporter.check_task_status.return_value = "DONE"
    module.transporter.release_xfer_handle.side_effect = [None, RuntimeError("cleanup failed"), None]
    module._update_task_status_once()
    assert recovery.fence.write_quiesced and recovery.fence.xfer_handle is None
    assert module.page_index_queue.empty()
    recovery.next_check_time = 0
    module._update_task_status_once()
    assert module.page_index_queue.get_nowait() == 0
    assert module.write_peer_kv_queue.empty()
    assert task.write_quiesced


def test_retired_status_query_exception_also_schedules_independent_recovery():
    module = _module(_PrefillTransModule)
    task = _task(src_page_index=0, dst_page_index=1, write_id="allocation-1", start_trans_time=0)
    task.write_submitted = True
    task.xfer_handle = 7
    module.waiting_dict[task.get_key()] = task
    module._update_task_status_once()
    module.transporter.check_task_status.side_effect = RuntimeError("query unavailable")
    module._update_task_status_once()
    assert task.write_uncertain
    assert module.write_peer_kv_queue.qsize() == 1
    assert module.write_recoveries[task.get_key()].original is task


def test_nixl_fence_uses_original_thread_and_prepared_connection_without_reconnect():
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter._write_contexts = {7: (threading.get_ident(), "old-src", "old-dst")}
    transporter.connect_add_remote_agent = MagicMock()
    transporter.nixl_agent = SimpleNamespace(
        query_xfer_backend=MagicMock(return_value="UCX"),
        make_prepped_xfer=MagicMock(return_value=8),
        transfer=MagicMock(),
    )
    task = _task(src_page_index=0, dst_page_index=1, xfer_handle=7)
    fence = replace(task, xfer_handle=None, write_submitted=False)
    transporter.start_write_fence(task, fence)
    transporter.nixl_agent.make_prepped_xfer.assert_called_once_with(
        "WRITE", "old-src", [0], "old-dst", [1], b"", backends=["UCX"]
    )
    assert fence.xfer_handle == 8 and fence.write_submitted
    transporter.connect_add_remote_agent.assert_not_called()
    transporter._write_contexts[7] = (-1, "old-src", "old-dst")
    with pytest.raises(AssertionError, match="original issuing thread"):
        transporter.start_write_fence(task, replace(fence, xfer_handle=None))
    assert transporter.nixl_agent.transfer.call_count == 1


def test_nixl_post_error_is_not_mistaken_for_an_unposted_handles_done_state():
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.remote_agents = {"D": object()}
    transporter._write_contexts = {}
    transporter.page_local_xfer_handles = "src"
    transporter.remote_agents["D"] = SimpleNamespace(page_xfer_handles="dst")
    transporter.nixl_agent = SimpleNamespace(
        make_prepped_xfer=MagicMock(return_value=7),
        transfer=MagicMock(return_value="ERR"),
        check_xfer_state=MagicMock(return_value="DONE"),
    )
    task = _task(src_page_index=0, dst_page_index=1)
    with pytest.raises(RuntimeError, match="submission returned ERR"):
        transporter.write_blocks_paged(task)
    assert task.write_submitted and task.xfer_handle == 7
    assert 7 in transporter._write_contexts
    transporter.nixl_agent.check_xfer_state.assert_not_called()


def test_nixl_query_exception_is_unknown_rather_than_native_err():
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.nixl_agent = SimpleNamespace(
        check_xfer_state=MagicMock(side_effect=RuntimeError("query result unavailable"))
    )
    task = _task(xfer_handle=7)
    assert transporter.check_task_status(task) == "UNKNOWN"
    transporter.nixl_agent.check_xfer_state.side_effect = None
    transporter.nixl_agent.check_xfer_state.return_value = "ERR"
    assert transporter.check_task_status(task) == "ERR"


@pytest.mark.skipif(nixl_kv_transporter.NixlWrapper is None, reason="NIXL bindings are not installed")
@pytest.mark.parametrize(
    "exception_name",
    [
        "nixlBackendError",
        "nixlRemoteDisconnectError",
        "nixlCancelledError",
        "nixlNotFoundError",
        "nixlUnknownError",
    ],
)
def test_native_binding_errors_traverse_real_python_wrapper_and_trigger_bounded_fence_retries(exception_name):
    # Keep the actual NIXL check_xfer_state body and actual native exception type;
    # replace only getXferStatus's native call, which needs no GPU for this test.
    native_error = getattr(nixl_kv_transporter.nixlBind, exception_name)
    agent = nixl_kv_transporter.NixlWrapper.__new__(nixl_kv_transporter.NixlWrapper)
    agent.agent = SimpleNamespace(getXferStatus=MagicMock(side_effect=native_error("injected native failure")))
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.nixl_agent = agent
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)
    module.transporter.check_task_status = transporter.check_task_status
    handles = []

    def submit(original, fence):
        handle = SimpleNamespace(_handle=len(handles) + 8)
        handles.append(handle)
        fence.xfer_handle = handle
        fence.write_submitted = True
        return "PROC"

    module.transporter.start_write_fence = submit
    for attempt in range(1, 4):
        recovery = _start_recovery(module)
        assert recovery.attempts == attempt
        recovery.next_check_time = 0
        module._update_task_status_once()
    assert recovery.fence.write_uncertain
    assert recovery.retired_handles == handles[:2]
    assert recovery.fence.xfer_handle is handles[2]
    assert agent.agent.getXferStatus.call_count == 3
    assert module.write_peer_kv_queue.empty()
    assert module.page_index_queue.empty() and not task.write_quiesced
    module.transporter.release_xfer_handle.assert_not_called()


@pytest.mark.parametrize("status", ["PROC", "UNKNOWN"])
def test_recovery_wait_has_deadline_and_retains_all_timed_out_handles(monkeypatch, status):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(prefill_trans_process.time, "monotonic", lambda: clock.now)
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)
    task.transfer_time_out_secs = 4
    handles = []

    def submit(original, fence):
        handle = len(handles) + 8
        handles.append(handle)
        fence.xfer_handle = handle
        fence.write_submitted = True
        return "PROC"

    module.transporter.start_write_fence = submit
    module.transporter.check_task_status.return_value = status
    for attempt in range(1, 4):
        recovery = _start_recovery(module)
        assert recovery.attempts == attempt
        assert recovery.deadline == clock.now + task.transfer_time_out_secs
        clock.now = recovery.deadline - 0.1
        recovery.next_check_time = 0
        module._update_task_status_once()
        assert recovery.post_confirmed
        assert module.write_peer_kv_queue.empty()
        clock.now = recovery.deadline
        recovery.next_check_time = 0
        module._update_task_status_once()
        assert not recovery.post_confirmed
    assert recovery.fence.write_uncertain
    assert recovery.retired_handles == [8, 9]
    assert recovery.fence.xfer_handle == 10
    assert module.write_peer_kv_queue.empty() and module.page_index_queue.empty()
    module.transporter.release_xfer_handle.assert_not_called()


def test_recovery_after_wait_timeout_drains_and_releases_previous_handle(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(prefill_trans_process.time, "monotonic", lambda: clock.now)
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)
    module.transporter.start_write_fence = _submit_fence
    recovery = _start_recovery(module)
    clock.now = recovery.deadline
    recovery.next_check_time = 0
    module._update_task_status_once()
    assert recovery.retired_handles == [8]
    assert not task.write_quiesced

    def submit(original, fence):
        fence.xfer_handle = 9
        fence.write_submitted = True
        return "PROC"

    module.transporter.start_write_fence = submit
    _start_recovery(module)
    module.transporter.check_task_status.return_value = "DONE"
    recovery.next_check_time = 0
    module._update_task_status_once()
    assert task.write_quiesced
    assert module.page_index_queue.get_nowait() == 0
    assert [call.args for call in module.transporter.release_xfer_handle.call_args_list] == [(9,), (8,), (7,)]


def test_completed_fence_proof_is_accepted_even_if_periodic_check_runs_after_deadline(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(prefill_trans_process.time, "monotonic", lambda: clock.now)
    module = _module(_PrefillTransModule)
    task = _uncertain_source(module)
    module.transporter.start_write_fence = _submit_fence
    recovery = _start_recovery(module)
    clock.now = recovery.deadline + 1
    module.transporter.check_task_status.return_value = "DONE"
    module._update_task_status_once()
    assert task.write_quiesced
    assert module.page_index_queue.get_nowait() == 0
    assert module.write_peer_kv_queue.empty()
    assert recovery.attempts == 1
