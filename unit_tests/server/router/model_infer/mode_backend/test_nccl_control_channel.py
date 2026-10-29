from __future__ import annotations

from typing import TYPE_CHECKING
import concurrent.futures
import socket
import threading

import rpyc

from lightllm.server.router.model_infer.mode_backend.pd import nccl_kv_transporter as module

if TYPE_CHECKING:
    import pytest


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
                executor.submit(module._NcclControlChannel, "127.0.0.1", first_port, first_port + 4) for _ in range(2)
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
