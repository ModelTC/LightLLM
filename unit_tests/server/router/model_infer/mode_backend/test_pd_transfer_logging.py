import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl import prefill_trans_process


@pytest.mark.parametrize("side, tokens", [("prefill", 3), ("decode", 3), ("decode", 0)])
def test_nixl_success_logging_keeps_processing_completed_pages(monkeypatch, side, tokens):
    transporter = NixlKVTransporter.__new__(NixlKVTransporter)
    transporter.nixl_agent = Mock()
    transporter._create_paged_xfer_handles = Mock()
    buffer = torch.empty((2, 3, 4, 5, 6), dtype=torch.float16)
    transporter._register_kv_move_buffer(buffer)
    assert transporter.page_len == buffer[0].numel() * buffer.element_size()
    transporter.release_xfer_handle = Mock()
    module = prefill_trans_process if side == "prefill" else decode_trans_process
    cls = module._PrefillTransModule if side == "prefill" else module._DecodeTransModule
    worker = cls.__new__(cls)
    worker.device_id = 0
    worker.transporter = transporter
    worker.page_index_queue = queue.Queue()
    worker.task_out_queue = queue.Queue()
    stats = Mock(interval=1)
    monkeypatch.setattr(module, "DebugStatsLogger", lambda *args: stats)
    monkeypatch.setattr(module, "logger", Mock())
    monkeypatch.setattr(module.torch.cuda, "set_device", lambda device: None)
    task = SimpleNamespace(
        src_page_index=0,
        dst_page_index=1,
        xfer_handle="handle",
        request_id=123,
        page_kind="kv",
        create_time=0,
        transfer_kv_num=lambda: tokens,
        createRetObj=lambda: SimpleNamespace(),
    )

    class StopLoop(Exception):
        pass

    items = iter([task, task, queue.Empty, StopLoop])

    def get(timeout):
        item = next(items)
        if isinstance(item, type):
            raise item()
        return item if side == "prefill" else (None, None, item)

    worker.success_queue = SimpleNamespace(get=get)
    with pytest.raises(StopLoop):
        worker.success_loop()
    assert worker.task_out_queue.qsize() == 2
    assert worker.page_index_queue.qsize() == 2
    assert transporter.release_xfer_handle.call_count == 2
    assert stats.add.call_count == (2 if tokens else 0)
    if tokens:
        assert stats.add.call_args.kwargs["bytes_total"] == transporter.page_len
    stats.flush.assert_called_once()


@pytest.mark.parametrize("side", ["prefill", "decode"])
@pytest.mark.parametrize("error_info", ["aborted req", "transfer timed out"])
def test_transfer_cancellation_is_debug_and_failures_are_error(monkeypatch, error_info, side):
    module = prefill_trans_process if side == "prefill" else decode_trans_process
    cls = module._PrefillTransModule if side == "prefill" else module._DecodeTransModule
    worker = cls.__new__(cls)
    worker.device_id = 0
    worker.page_index_queue = queue.Queue()
    worker.task_out_queue = queue.Queue()
    worker.transporter = Mock()
    worker._abort = Mock()
    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module.torch.cuda, "set_device", lambda _: None)
    task = SimpleNamespace(
        src_page_index=2,
        dst_page_index=2,
        xfer_handle=None,
        request_id=123,
        error_info=error_info,
        createRetObj=lambda: "result",
    )

    class StopLoop(Exception):
        pass

    worker.failed_queue = Mock()
    worker.failed_queue.get.side_effect = [task, StopLoop()]
    with pytest.raises(StopLoop):
        worker.fail_loop()
    worker._abort.assert_called_once_with(request_id=123, error_info=error_info)
    notify = (
        worker.transporter.send_error_info_to_decode_node
        if side == "prefill"
        else worker.transporter.send_error_info_to_prefill_node
    )
    notify.assert_called_once_with(trans_task=task)
    assert worker.page_index_queue.get_nowait() == 2
    assert worker.task_out_queue.get_nowait() == "result"
    if error_info == "aborted req":
        logger.error.assert_not_called()
        logger.debug.assert_called_once_with("KV transfer cancelled: %s", "result")
    else:
        logger.error.assert_called_once_with("KV transfer failed: %s", "result")


def test_missing_nixl_fails_before_initializing_transport(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend.pd import nixl_kv_transporter, kv_transporter

    monkeypatch.setattr(nixl_kv_transporter, "NixlWrapper", None)
    with pytest.raises(ImportError, match="pd_trans_mode=nixl requires NIXL; install NIXL"):
        NixlKVTransporter(node_id=1, tp_idx=0, kv_move_buffer=None)
    with pytest.raises(ImportError, match="select pd_trans_mode=nccl"):
        kv_transporter.create_kv_transporter(SimpleNamespace(pd_trans_mode="nixl"), 1, 0, None)


def test_nixl_factory_forwards_transport_arguments(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend.pd import nixl_kv_transporter, kv_transporter

    constructor = Mock()
    monkeypatch.setattr(nixl_kv_transporter, "NixlKVTransporter", constructor)
    buffer = object()
    result = kv_transporter.create_kv_transporter(SimpleNamespace(pd_trans_mode="nixl"), 7, 2, buffer)
    constructor.assert_called_once_with(node_id=7, tp_idx=2, kv_move_buffer=buffer)
    assert result is constructor.return_value


@pytest.mark.parametrize("level", [20, 10])
@pytest.mark.parametrize("kind,tokens", [("kv", 3), ("kv", 0), ("state", 0)])
def test_decode_success_real_debug_window(monkeypatch, caplog, level, kind, tokens):
    import logging
    from lightllm.utils import log_utils

    module = decode_trans_process
    worker = module._DecodeTransModule.__new__(module._DecodeTransModule)
    worker.device_id = 0
    worker.page_index_queue = queue.Queue()
    worker.task_out_queue = queue.Queue()
    worker.transporter = Mock(kv_move_buffer=torch.empty((1, 4), dtype=torch.float16))
    logger = logging.getLogger("decode-transfer-regression")
    monkeypatch.setattr(module, "logger", logger)
    monkeypatch.setattr(module.torch.cuda, "set_device", Mock())
    clock = [0]
    monkeypatch.setattr(log_utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(log_utils, "get_log_stats_interval", lambda: 10)
    transfer = Mock(return_value=tokens)
    task = SimpleNamespace(
        request_id=123,
        page_kind=kind,
        transfer_kv_num=transfer,
        dst_page_index=1,
        xfer_handle=None,
        create_time=0,
        createRetObj=lambda: "done",
    )
    start, end = Mock(), Mock()
    start.elapsed_time.return_value = 2.5
    count = [0]

    class StopLoop(BaseException):
        pass

    def get(timeout):
        assert timeout == 10
        count[0] += 1
        if count[0] <= 3:
            return end, start, task
        if count[0] == 4:
            clock[0] = 11
            raise queue.Empty()
        raise StopLoop()

    worker.success_queue = SimpleNamespace(get=get)
    with caplog.at_level(level, logger=logger.name), pytest.raises(StopLoop):
        worker.success_loop()
    assert worker.task_out_queue.qsize() == 3 and worker.page_index_queue.qsize() == 3
    assert end.synchronize.call_count == 3
    records = [r for r in caplog.records if r.name == logger.name]
    assert len(records) == 2 * int(level == 10 and (kind != "kv" or tokens > 0))
    if level == 20:
        transfer.assert_not_called()
        start.elapsed_time.assert_not_called()
    if records:
        record = records[1]
        assert record.event == "kv.transfer.decode"
        assert record.pages == 2 and record.kv_tokens_total == tokens * 2
        assert record.state_pages == (2 if kind == "state" else 0)
        assert record.read_page_gpu_ms_total == 5
        assert record.bytes_total == 16
        assert record.request_id_sample == {"count": 1, "sample": [123]}


def test_prefill_idle_flushes_transfer_and_telemetry_separately(monkeypatch):
    module = prefill_trans_process
    worker = module._PrefillTransModule.__new__(module._PrefillTransModule)
    worker.device_id = 0
    worker._debug_telemetry = Mock()
    transfer = Mock(interval=7)
    monkeypatch.setattr(module, "DebugStatsLogger", Mock(return_value=transfer))
    monkeypatch.setattr(module.torch.cuda, "set_device", Mock())

    class StopLoop(BaseException):
        pass

    worker.success_queue = Mock()
    worker.success_queue.get.side_effect = [queue.Empty(), StopLoop()]
    with pytest.raises(StopLoop):
        worker.success_loop()
    assert worker.success_queue.get.call_args.kwargs == {"timeout": 7}
    transfer.flush.assert_called_once()
    worker._debug_telemetry.flush.assert_called_once()


def test_prefill_info_does_not_collect_nixl_telemetry(monkeypatch):
    import logging
    import threading

    worker = prefill_trans_process._PrefillTransModule.__new__(prefill_trans_process._PrefillTransModule)
    task = Mock(xfer_handle="handle")
    task.get_key.return_value = "key"
    worker.waiting_dict = {"key": task}
    worker.waiting_dict_lock = threading.Lock()
    worker.transporter = Mock(capture_telemetry=True)
    worker.transporter.check_task_status.return_value = "DONE"
    worker.success_queue = queue.Queue()
    logger = logging.getLogger("telemetry-info-regression")
    monkeypatch.setattr(prefill_trans_process, "logger", logger)
    old = logger.level
    logger.setLevel(logging.INFO)
    try:
        worker._update_task_status_once()
    finally:
        logger.setLevel(old)
    worker.transporter.nixl_agent.get_xfer_telemetry.assert_not_called()
    assert worker.success_queue.get_nowait() is task
