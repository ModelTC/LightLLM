import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.pd.nixl_kv_transporter import NixlKVTransporter
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl import prefill_trans_process


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_nixl_success_logging_keeps_processing_completed_pages(monkeypatch, side):
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
        transfer_kv_num=lambda: 3,
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
    assert stats.add.call_count == 2
    assert stats.add.call_args.kwargs["bytes_total"] == transporter.page_len
    stats.flush.assert_called_once()
