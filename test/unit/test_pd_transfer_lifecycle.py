import queue
import threading
from types import SimpleNamespace

from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_trans_process import (
    _DecodeTransModule,
)
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_trans_process import (
    _PrefillTransModule,
)


class _RecordingQueue:
    def __init__(self, events):
        self.events = events

    def put(self, value):
        self.events.append(("recycle", value))


class _FakeTransporter:
    def __init__(self, status, events, release_error=None, notify_errors=None):
        self.status = status
        self.events = events
        self.release_error = release_error
        self.notify_errors = list(notify_errors or [])

    def check_task_status(self, trans_task):
        self.events.append(("check", trans_task.xfer_handle))
        return self.status

    def release_xfer_handle(self, handle):
        self.events.append(("release", handle))
        if self.release_error is not None:
            raise self.release_error

    def send_error_info_to_decode_node(self, trans_task):
        self.events.append(("notify", trans_task.transfer_quiesced))
        if self.notify_errors:
            raise self.notify_errors.pop(0)


def _make_task():
    return SimpleNamespace(
        dst_page_index=7,
        error_info="timeout",
        src_page_index=3,
        transfer_quiesced=False,
        xfer_handle=11,
        get_key=lambda: "task-key",
        to_str=lambda: "task-key",
    )


def test_prefill_keeps_source_page_while_transfer_is_in_progress():
    events = []
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.transporter = _FakeTransporter("PROC", events)
    module.page_index_queue = _RecordingQueue(events)
    task = _make_task()

    status = module._try_finish_failed_xfer_drain(task)

    assert status is None
    assert events == [("check", 11)]
    assert task.src_page_index == 3
    assert task.xfer_handle == 11
    assert not task.transfer_quiesced


def test_prefill_recycles_source_page_only_after_terminal_status_and_release():
    events = []
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.transporter = _FakeTransporter("DONE", events)
    module.page_index_queue = _RecordingQueue(events)
    task = _make_task()

    status = module._try_finish_failed_xfer_drain(task)

    assert status == "DONE"
    assert events == [
        ("check", 11),
        ("release", 11),
        ("recycle", 3),
        ("notify", True),
    ]
    assert task.src_page_index is None
    assert task.xfer_handle is None
    assert task.transfer_quiesced


def test_prefill_keeps_page_quarantined_when_handle_release_fails():
    events = []
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.transporter = _FakeTransporter("ERR", events, release_error=RuntimeError("still active"))
    module.page_index_queue = _RecordingQueue(events)
    task = _make_task()

    try:
        module._try_finish_failed_xfer_drain(task)
    except RuntimeError as exc:
        assert str(exc) == "still active"
    else:
        raise AssertionError("release failure must be propagated to the drain loop")

    assert events == [("check", 11), ("release", 11)]
    assert task.src_page_index == 3
    assert task.xfer_handle == 11
    assert not task.transfer_quiesced


def test_prefill_retries_quiesced_ack_without_rechecking_released_handle():
    events = []
    module = _PrefillTransModule.__new__(_PrefillTransModule)
    module.transporter = _FakeTransporter(
        "DONE",
        events,
        notify_errors=[RuntimeError("peer unavailable")],
    )
    module.page_index_queue = _RecordingQueue(events)
    task = _make_task()

    try:
        module._try_finish_failed_xfer_drain(task)
    except RuntimeError as exc:
        assert str(exc) == "peer unavailable"
    else:
        raise AssertionError("notify failure must be propagated to the drain loop")

    assert task.transfer_quiesced
    assert task.xfer_handle is None
    assert task.src_page_index is None

    status = module._try_finish_failed_xfer_drain(task)

    assert status == "QUIESCED"
    assert events == [
        ("check", 11),
        ("release", 11),
        ("recycle", 3),
        ("notify", True),
        ("notify", True),
    ]


def test_decode_quarantines_destination_page_until_quiesced_ack():
    module = _DecodeTransModule.__new__(_DecodeTransModule)
    module.waiting_dict_lock = threading.Lock()
    module.quarantined_dict = {}
    module.failed_queue = queue.Queue()
    module.page_index_queue = queue.Queue()
    task = _make_task()

    module._queue_failed_task(task)

    assert module.quarantined_dict == {"task-key": task}
    assert module.failed_queue.get_nowait() is task
    assert module.page_index_queue.empty()

    quarantined_task = module.quarantined_dict.pop(task.get_key())
    module._recycle_quarantined_page(quarantined_task)

    assert module.page_index_queue.get_nowait() == 7
    assert task.dst_page_index is None
    assert task.transfer_quiesced
