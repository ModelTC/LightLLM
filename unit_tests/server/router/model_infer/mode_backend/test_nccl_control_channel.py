import errno
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from rpyc.utils.server import ThreadedServer

from lightllm.server.router.model_infer.mode_backend.pd import nccl_kv_transporter


def available_port_pair():
    for _ in range(100):
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port == 65535:
                continue
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port
    raise RuntimeError("could not find two available ports")


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux SO_REUSEADDR bind behavior")
@pytest.mark.parametrize("extra_ports", [0, 1], ids=["exhausted", "retry"])
def test_control_channels_contending_for_port(monkeypatch, extra_ports):
    port = available_port_pair()
    bound = threading.Barrier(2)
    servers, server_threads, channels, errors = [], [], [], []

    class ContendedServer(ThreadedServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.append(self)
            if self.port == port:
                bound.wait(timeout=5)

        def start(self):
            server_threads.append(threading.current_thread())
            super().start()

    monkeypatch.setattr(nccl_kv_transporter, "ThreadedServer", ContendedServer)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(nccl_kv_transporter._NcclControlChannel, "127.0.0.1", port, port + extra_ports)
                for _ in range(2)
            ]
            for future in futures:
                try:
                    channels.append(future.result(timeout=10))
                except RuntimeError as error:
                    errors.append(error)

        assert len(channels) == 1 + extra_ports
        assert {channel.port for channel in channels} == set(range(port, port + extra_ports + 1))
        assert len(errors) == 1 - extra_ports
        for error in errors:
            assert isinstance(error.__cause__, OSError)
            assert error.__cause__.errno == errno.EADDRINUSE
        failed = [server for server in servers if server not in [channel._server for channel in channels]]
        assert len(failed) == 1
        assert failed[0].listener.fileno() == -1
        for index, sender in enumerate(channels):
            receiver = channels[(index + 1) % len(channels)]
            assert receiver._server.active
            sender.send_notif(str(index), "127.0.0.1", receiver.port, b"page ready")
            assert receiver.get_notifs() == [b"page ready"]
            sender.send_store_value(str(index), "127.0.0.1", receiver.port, "bootstrap", b"communicator")
            assert receiver.wait_store_value("bootstrap", timeout=1) == b"communicator"
    finally:
        for channel in channels:
            channel.close()
        for server in servers:
            server.close()
        for thread in server_threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
