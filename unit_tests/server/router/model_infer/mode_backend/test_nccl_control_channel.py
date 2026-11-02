from __future__ import annotations

import concurrent.futures
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import rpyc
import pytest

from lightllm.server.router.model_infer.mode_backend.pd import nccl_kv_transporter as module


def test_page_events_are_recorded_after_send_and_receive(monkeypatch: pytest.MonkeyPatch):
    page_buffers = module.torch.empty(size=(2, 8), dtype=module.torch.uint8, device="cpu")
    transporter = SimpleNamespace(kv_move_buffer=page_buffers, args=SimpleNamespace(pd_detail_log=False))
    peer = module._NcclPeer(transporter, "remote")
    stream = object()
    operations = []
    comm = SimpleNamespace(
        send=Mock(side_effect=lambda tensor, dst, stream: operations.append(("send", tensor.numel(), dst, stream))),
        recv=Mock(side_effect=lambda tensor, src, stream: operations.append(("recv", tensor.numel(), src, stream))),
    )
    monkeypatch.setattr(peer, "_ensure_comm", lambda is_server: comm)
    monkeypatch.setattr(peer, "_get_stream", lambda: stream)
    event = Mock()
    event.record.side_effect = lambda stream: operations.append(("record", stream))
    monkeypatch.setattr(module.torch.cuda, "Event", lambda: event)
    task = SimpleNamespace(src_page_index=1, dst_page_index=0)

    handle = peer.send_page(task)
    assert operations == [("send", 8, 1, stream), ("record", stream)]
    assert handle.mode == "send"
    assert handle.finish_cuda_event is event
    event.query.return_value = True
    assert handle.check_status() == "DONE"

    operations.clear()
    receive_queue = Mock()
    monkeypatch.setattr(peer, "_get_recv_queue", lambda: receive_queue)
    peer.start_recv(task)
    queued_task = receive_queue.put.call_args.args[0]
    assert queued_task.xfer_handle is task.xfer_handle
    assert task.xfer_handle.mode == "recv"
    assert task.xfer_handle.finish_cuda_event is None
    peer._recv_page(queued_task)
    assert operations == [("recv", 8, 0, stream), ("record", stream)]
    assert task.xfer_handle.finish_cuda_event is event
    assert task.xfer_handle.check_status() == "DONE"
    assert event.query.call_count == 2


def test_receive_waits_for_record_to_return(monkeypatch: pytest.MonkeyPatch):
    record_started = threading.Event()
    finish_record = threading.Event()
    event = Mock()

    def record(stream: object) -> None:
        record_started.set()
        assert finish_record.wait(timeout=5)

    event.record.side_effect = record
    monkeypatch.setattr(module.torch.cuda, "Event", lambda: event)
    transporter = SimpleNamespace(
        kv_move_buffer=module.torch.empty(size=(1, 8), dtype=module.torch.uint8),
        args=SimpleNamespace(pd_detail_log=False),
    )
    peer = module._NcclPeer(transporter, "remote")
    monkeypatch.setattr(peer, "_ensure_comm", lambda is_server: SimpleNamespace(recv=Mock()))
    monkeypatch.setattr(peer, "_get_stream", object)
    handle = module._NcclXferHandle(peer_name="remote", mode="recv")
    task = SimpleNamespace(dst_page_index=0, xfer_handle=handle)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        receive = executor.submit(peer._recv_page, task)
        try:
            assert record_started.wait(timeout=5)
            assert handle.check_status() == "PROC"
            assert handle.finish_cuda_event is None
            event.query.assert_not_called()
        finally:
            finish_record.set()
        receive.result(timeout=5)
        assert handle.check_status() == "DONE"
    assert handle.finish_cuda_event is event
    event.query.assert_called_once()


def test_receive_submission_failure_wakes_waiter(monkeypatch: pytest.MonkeyPatch):
    transporter = SimpleNamespace(
        kv_move_buffer=module.torch.empty(size=(1, 8), dtype=module.torch.uint8),
        send_error_info_to_prefill_node=Mock(),
    )
    peer = module._NcclPeer(transporter, "remote")
    comm = SimpleNamespace(recv=Mock(side_effect=RuntimeError("recv failed")))
    monkeypatch.setattr(peer, "_ensure_comm", lambda is_server: comm)
    monkeypatch.setattr(peer, "_get_stream", object)
    monkeypatch.setattr(peer, "_drop_comm", Mock())
    handle = module._NcclXferHandle(peer_name="remote", mode="recv")
    task = SimpleNamespace(dst_page_index=0, xfer_handle=handle)

    peer._recv_page(task)

    assert handle.check_status() == "ERR"
    assert handle.error_info == "recv failed"
    transporter.send_error_info_to_prefill_node.assert_called_once_with(task)


def test_concurrent_control_channels_retry_port_before_publishing_metadata(monkeypatch: pytest.MonkeyPatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        first_port = probe.getsockname()[1]

    original = module.ThreadedServer
    both_bound = threading.Barrier(2)

    def bind_before_listening(*args: object, **kwargs: object):
        server = original(*args, **kwargs)
        if kwargs["port"] == first_port:
            # Force both SO_REUSEADDR sockets to bind before either listens.
            both_bound.wait(timeout=10)
        return server

    monkeypatch.setattr(module, "ThreadedServer", bind_before_listening)
    channels = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    module._NcclControlChannel,
                    "127.0.0.1",
                    first_port,
                    first_port + 4,
                )
                for _ in range(2)
            ]
            for future in futures:
                channels.append(future.result(timeout=15))
        assert len({channel.port for channel in channels}) == 2
        for index, channel in enumerate(channels):
            payload = f"channel-{index}".encode()
            connection = rpyc.connect("127.0.0.1", channel.port)
            try:
                connection.root.push_notif(payload)
            finally:
                connection.close()
            assert channel.get_notifs() == [payload]
    finally:
        for channel in channels:
            channel.close()
