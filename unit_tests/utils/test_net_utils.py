import errno
import socket

import pytest

from lightllm.utils.net_utils import validate_ports


def test_validate_ports_rejects_duplicates():
    with pytest.raises(RuntimeError, match="conflicting ports"):
        validate_ports([18111, 18111])


def test_validate_ports_rejects_an_active_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(RuntimeError, match="has been used"):
            validate_ports([listener.getsockname()[1]])


def test_validate_ports_allows_restart_after_server_closes_a_connection():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(2)
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
                client.shutdown(socket.SHUT_WR)
                assert connection.recv(1) == b""

    with socket.socket() as probe:
        with pytest.raises(OSError) as error:
            probe.bind(("", port))
        assert error.value.errno == errno.EADDRINUSE
    validate_ports([port])
