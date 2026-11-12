import errno
import socket
import sys

import pytest

from lightllm.utils.net_utils import validate_ports


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux TCP port reuse semantics")


def test_validate_ports_rejects_duplicates():
    with pytest.raises(RuntimeError, match="conflicting ports"):
        validate_ports([0, 0])


def test_validate_ports_rejects_active_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match=f"port {port} has been used"):
            validate_ports([port])


def test_validate_ports_allows_restart_after_closed_connection():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(2)
        address = listener.getsockname()
        with socket.create_connection(address, timeout=2) as client:
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""

    with socket.socket() as probe:
        with pytest.raises(OSError) as error:
            probe.bind(("", address[1]))
        assert error.value.errno == errno.EADDRINUSE
    validate_ports([address[1]])
