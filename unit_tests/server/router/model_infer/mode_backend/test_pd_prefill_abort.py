import queue
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lightllm.server.pd_io_struct import PDAbortReq, PDChunckedTransTask
from lightllm.server.router.model_infer.mode_backend import base_backend
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl import PDDecodeNode
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import _DecodeTransModule
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


def _make_transfer_task(page_kind="kv"):
    return PDChunckedTransTask(
        request_id=123,
        start_kv_index=0 if page_kind == "kv" else 4,
        end_kv_index=4,
        request_kv_len=4,
        time_out_secs=10,
        pd_master_node_id=0,
        prefill_dp_index=0,
        decode_dp_index=0,
        src_device_id=0,
        dst_device_id=0,
        mem_indexes=list(range(4)) if page_kind == "kv" else [],
        prefill_agent_name="P",
        prefill_agent_metadata=None,
        prefill_num_pages=None,
        prefill_page_reg_desc=None,
        decode_agent_name="D",
        decode_agent_metadata=None,
        decode_num_pages=None,
        decode_page_reg_desc=None,
        first_gen_token_id=None,
        first_gen_token_logprob=None,
        page_kind=page_kind,
    )


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


@pytest.mark.parametrize("success_num,failed_num", [(2, 0), (1, 1), (0, 2)])
@pytest.mark.parametrize("abort_while_waiting", [False, True])
def test_prefill_capacity_pause_waits_for_pending_pd_tasks(monkeypatch, success_num, failed_num, abort_while_waiting):
    req = SimpleNamespace(
        req_id=123,
        infer_aborted=False,
        pd_task_num=2,
        pd_task_failed_num=0,
        pd_task_success_num=0,
        pd_trans_device_id=0,
        cur_kv_len=8192,
        shm_req=SimpleNamespace(input_len=8193),
        filter_mark=False,
        wait_pause=False,
        paused=False,
        finish_status=SimpleNamespace(is_finished=lambda: False),
        cpu_cache_task_status=SimpleNamespace(is_not_started=lambda: True),
        is_slave_req=lambda: False,
        prefill_need_token_num=lambda **kwargs: (1, 1),
    )
    backend = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    backend.args = SimpleNamespace(enable_cpu_cache=False, enable_prefill_decode_mixed=False)
    backend.is_master_in_dp = True
    backend.is_deepseek_v4 = True
    backend.support_overlap = False
    backend.disable_chunked_prefill = False
    backend.batch_max_tokens = 8192
    backend.info_queue = queue.Queue()
    backend._timer_merge_radix_tree = MagicMock()
    backend._reorder_pd_high_priority_reqs = lambda reqs: reqs
    backend._reorder_long_prefill_reqs = lambda reqs: reqs

    context = prefill_impl.g_infer_context
    monkeypatch.setattr(context, "requests_mapping", {req.req_id: req})
    monkeypatch.setattr(context, "get_can_alloc_token_num", lambda: 0)
    monkeypatch.setattr(context, "get_can_alloc_dsv4_swa_page_num", lambda: 0)
    monkeypatch.setattr(context, "cache_placement_controller", SimpleNamespace(set_req_cache_way=MagicMock()))
    monkeypatch.setattr(context, "filter_reqs", MagicMock())
    pause_reqs = MagicMock()
    monkeypatch.setattr(context, "pause_reqs", pause_reqs)

    # Capacity pressure marks the unfinished request for pausing on the next iteration.
    assert backend._get_classed_reqs(req_ids=[req.req_id], no_decode=True) == ([], [])
    assert req.wait_pause
    pause_reqs.assert_called_once_with([], is_master_in_dp=True)

    req.infer_aborted = abort_while_waiting
    pause_reqs.reset_mock()
    assert backend._get_classed_reqs(req_ids=[req.req_id], no_decode=True) == ([], [])
    pause_reqs.assert_called_once_with([], is_master_in_dp=True)

    req.pd_task_success_num = success_num
    req.pd_task_failed_num = failed_num
    pause_reqs.reset_mock()
    context.filter_reqs.reset_mock()
    backend._get_classed_reqs(req_ids=[req.req_id], no_decode=True)
    if abort_while_waiting:
        pause_reqs.assert_called_once_with([], is_master_in_dp=True)
        context.filter_reqs.assert_called_once_with(finished_reqs=[req])
        assert backend.info_queue.qsize() == 2
    else:
        pause_reqs.assert_called_once_with([req], is_master_in_dp=True)
        assert backend.info_queue.empty()


def test_prefill_keeps_chunk_overlap_while_pd_tasks_are_pending(monkeypatch):
    req = SimpleNamespace(
        req_id=123,
        infer_aborted=False,
        wait_pause=False,
        pd_task_num=2,
        pd_task_failed_num=0,
        pd_task_success_num=0,
        cur_kv_len=8192,
        shm_req=SimpleNamespace(input_len=8193),
    )
    monkeypatch.setattr(prefill_impl.g_infer_context, "requests_mapping", {req.req_id: req})
    backend = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    backend.is_master_in_dp = True
    backend.info_queue = queue.Queue()

    assert backend._filter_not_ready_reqs([req.req_id]) == [req]
    assert backend.info_queue.empty()


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
    module.use_write_quiescence = False
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
    module.use_write_quiescence = False
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


@pytest.mark.parametrize("release_fails", [False, True])
def test_prefill_failed_transfer_releases_handle_before_reusing_page(monkeypatch, release_fails):
    task = SimpleNamespace(
        src_page_index=2,
        xfer_handle=object(),
        error_info=None,
        createRetObj=MagicMock(return_value=object()),
    )
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.use_write_quiescence = False
    module.device_id = 0
    module.failed_queue = _SequenceQueue(task)
    module.page_index_queue = queue.Queue()
    module.task_out_queue = queue.Queue()
    monkeypatch.setattr(prefill_trans_process.torch.cuda, "set_device", lambda _device_id: None)

    def release(handle):
        assert handle is task.xfer_handle
        assert module.page_index_queue.empty()
        assert module.task_out_queue.empty()
        if release_fails:
            raise RuntimeError("transfer cancellation failed")

    module.transporter = SimpleNamespace(release_xfer_handle=MagicMock(side_effect=release))
    if release_fails:
        with pytest.raises(RuntimeError, match="transfer cancellation failed"):
            module.fail_loop()
        assert module.page_index_queue.empty()
        assert module.task_out_queue.empty()
        task.createRetObj.assert_not_called()
    else:
        with pytest.raises(_StopLoop):
            module.fail_loop()
        assert module.page_index_queue.get_nowait() == 2
        assert module.task_out_queue.get_nowait() is task.createRetObj.return_value
    module.transporter.release_xfer_handle.assert_called_once()


@pytest.mark.parametrize("side", ["prefill", "decode"])
@pytest.mark.parametrize("outcome", ["send_error", "timeout", "advanced"])
def test_notify_send_error_only_cleans_up_the_task_it_still_owns(monkeypatch, side, outcome):
    task = _make_transfer_task()
    task.time_out = lambda: True
    is_prefill = side == "prefill"
    module_class = _PrefillTransModule if is_prefill else _DecodeTransModule
    module = module_class.__new__(module_class)
    module.use_write_quiescence = False
    module.device_id = 0
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {}
    module.request_last_progress_time = {}
    module.failed_queue = queue.Queue()
    next_stage = queue.Queue()
    monkeypatch.setattr(prefill_trans_process.torch.cuda, "set_device", lambda _device_id: None)

    def send(*args, **kwargs):
        if outcome == "timeout":
            module._check_tasks_time_out()
        elif outcome == "advanced":
            with module.waiting_dict_lock:
                next_stage.put(module.waiting_dict.pop(task.get_key()))
        raise RuntimeError("notification failed after another thread took ownership")

    module.transporter = SimpleNamespace(
        send_write_request_task_to_decode_node=send,
        send_write_ready_task_to_prefill_node=send,
        remove_remote_agent=MagicMock(),
        send_error_info_to_decode_node=MagicMock(),
        send_error_info_to_prefill_node=MagicMock(),
    )
    if is_prefill:
        task.src_page_index = 2
        module.ready_transfer_queue = _SequenceQueue((SimpleNamespace(synchronize=lambda: None), task))
        loop = module.ready_transfer_loop
    else:
        module.page_index_queue = _SequenceQueue(7)
        module.request_page_task_queue = _SequenceQueue(task)
        loop = module.request_page_loop
    with pytest.raises(_StopLoop):
        loop()

    if outcome == "send_error":
        module.transporter.remove_remote_agent.assert_called_once()
    else:
        module.transporter.remove_remote_agent.assert_not_called()
    if outcome == "advanced":
        assert module.failed_queue.empty()
        assert next_stage.get_nowait() is task
        assert task.error_info is None
        return

    # Only the winning owner cleans up: one page release and one receipt.
    assert module.failed_queue.qsize() == 1
    module.failed_queue = _SequenceQueue(module.failed_queue.get_nowait())
    module.page_index_queue = queue.Queue()
    module.task_out_queue = queue.Queue()
    with pytest.raises(_StopLoop):
        module.fail_loop()
    assert module.page_index_queue.qsize() == module.task_out_queue.qsize() == 1


def test_prefill_send_error_does_not_remove_a_write_that_already_started(monkeypatch):
    task = _make_transfer_task()
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.use_write_quiescence = False
    module.device_id = 0
    module.waiting_dict_lock = threading.Lock()
    module.waiting_dict = {}
    module.failed_queue = queue.Queue()
    module.ready_transfer_queue = _SequenceQueue((SimpleNamespace(synchronize=lambda: None), task))
    monkeypatch.setattr(prefill_trans_process.torch.cuda, "set_device", lambda _device_id: None)

    def send(_task):
        # The peer received the notify and WRITE started before the send call reported an error.
        with module.waiting_dict_lock:
            module.waiting_dict.pop(task.get_key())
            task.xfer_handle = 7
            module.waiting_dict[task.get_key()] = task
        raise RuntimeError("late notification error")

    module.transporter = SimpleNamespace(send_write_request_task_to_decode_node=send, remove_remote_agent=MagicMock())
    with pytest.raises(_StopLoop):
        module.ready_transfer_loop()
    assert module.waiting_dict == {task.get_key(): task}
    assert module.failed_queue.empty()
    module.transporter.remove_remote_agent.assert_not_called()


@pytest.mark.parametrize("first_failed", [False, True])
def test_pd_receipts_are_counted_once_per_task_across_batches(monkeypatch, first_failed):
    kv_task = replace(_make_transfer_task(), start_kv_index=4, mem_indexes=[])
    state_task = _make_transfer_task(page_kind="att_state")
    kv_task.error_info = "failed" if first_failed else None
    state_task.error_info = "failed"
    first, second = kv_task.createRetObj(), state_task.createRetObj()
    req = SimpleNamespace(pd_task_num=2, pd_task_failed_num=0, pd_task_success_num=0, pd_task_finished_keys=set())
    monkeypatch.setattr(base_backend.g_infer_context, "requests_mapping", {kv_task.request_id: req})
    backend = PDChunkedPrefillForPrefillNode.__new__(PDChunkedPrefillForPrefillNode)
    backend.is_pd_decode_mode = False
    backend.logger = MagicMock()
    backend.shm_pd_trans_io_buffer = SimpleNamespace(
        read_obj=MagicMock(
            side_effect=[[first, replace(first)], [replace(first, has_error=not first_failed), second, second]]
        ),
        sub_state=MagicMock(),
    )

    backend._read_pd_trans_io_buffer_and_update_req_status()
    assert req.pd_task_failed_num + req.pd_task_success_num == 1
    backend._read_pd_trans_io_buffer_and_update_req_status()
    assert req.pd_task_failed_num == int(first_failed) + 1
    assert req.pd_task_success_num == int(not first_failed)
    assert req.pd_task_finished_keys == {kv_task.get_key(), state_task.get_key()}


def test_pd_receipt_preserves_page_kind_identity():
    for kind in ("kv", "att_state"):
        task = _make_transfer_task(page_kind=kind)
        assert task.createRetObj().get_key() == task.get_key()
