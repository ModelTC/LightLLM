import socket
import types

import pytest
import rpyc

from lightllm.server.router.profiler_service import RouterProfilerCmdQueue, start_router_profiler_server


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def profiler_server(monkeypatch):
    port = _free_port()
    from lightllm.utils import shm_port_args

    monkeypatch.setattr(shm_port_args, "get_shm_port_args", lambda: types.SimpleNamespace(router_profiler_port=port))
    queue = RouterProfilerCmdQueue()
    server, _ = start_router_profiler_server(types.SimpleNamespace(enable_profiling="torch_profiler"), queue)
    yield server, port, queue
    server.close()


def test_profiler_server_binds_loopback_only(profiler_server):
    server, _port, _queue = profiler_server
    assert server.listener.getsockname()[0] == "127.0.0.1"


def test_profiler_server_accepts_known_cmds(profiler_server):
    _server, port, queue = profiler_server
    conn = rpyc.connect("127.0.0.1", port)
    conn.root.profiler_cmd("start")
    assert queue.pop() == "start"
    conn.root.profiler_cmd("stop")
    assert queue.pop() == "stop"
    conn.close()


def test_profiler_server_rejects_unknown_cmd(profiler_server):
    _server, port, queue = profiler_server
    conn = rpyc.connect("127.0.0.1", port)
    with pytest.raises(Exception):
        conn.root.profiler_cmd("drop_tables")
    assert queue.pop() is None
    conn.close()


def test_disabled_profiling_starts_nothing():
    assert start_router_profiler_server(types.SimpleNamespace(enable_profiling=None), None) == (None, None)
