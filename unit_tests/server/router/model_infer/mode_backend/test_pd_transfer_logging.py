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
