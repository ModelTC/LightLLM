import asyncio
from datetime import timedelta
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
import torch.distributed as dist

from lightllm.server.config_server import nccl_tcp_store
from lightllm.utils import process_check


def _run_store(connection: Connection, port: int) -> None:
    process = nccl_tcp_store.start_tcp_store_server(nccl_store_host="127.0.0.1", nccl_store_port=port)
    try:
        connection.send(process.pid)
        connection.recv()
    finally:
        process.terminate()
        process.wait(timeout=10)
        connection.close()


@pytest.mark.skipif(sys.platform != "linux", reason="requires fork")
@pytest.mark.parametrize("daemon", [False, True])
def test_tcp_store_serves_clients_from_a_daemon_worker(daemon: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name="LIGHTLLM_UNIQUE_SERVICE_NAME_ID", value="tcp_store_regression")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = mp.get_context(method="fork")
    reader, writer = context.Pipe()
    worker = context.Process(target=_run_store, args=(writer, port), daemon=daemon)
    worker.start()
    server_pid = None
    try:
        assert reader.poll(timeout=10), "worker did not start TCPStore"
        server_pid = reader.recv()
        client = dist.TCPStore(
            host_name="127.0.0.1", port=port, is_master=False, wait_for_workers=False, timeout=timedelta(seconds=20)
        )
        client.set("roundtrip", "value")
        assert client.get("roundtrip") == b"value"
        del client
        reader.send("stop")
        worker.join(timeout=10)
        assert worker.exitcode == 0
        assert not process_check.is_process_active(pid=server_pid)
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)
        if server_pid is not None and process_check.is_process_active(pid=server_pid):
            psutil.Process(pid=server_pid).kill()
        worker.close()
        reader.close()
        writer.close()


def test_restart_reaps_previous_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target=process_check, name="start_parent_check_thread", value=lambda: None)
    from lightllm.server.config_server import api_http

    old = Mock(spec=subprocess.Popen)
    new = Mock(spec=subprocess.Popen)
    start = Mock(return_value=new)
    monkeypatch.setattr(
        target=api_http, name="get_env_start_args", value=lambda: SimpleNamespace(config_server_host="host")
    )
    monkeypatch.setattr(target=api_http, name="start_tcp_store_server", value=start)
    monkeypatch.setattr(target=api_http, name="global_store_port_to_process", value={29501: old})
    monkeypatch.setattr(target=api_http, name="global_store_port_to_client_states", value={})
    monkeypatch.setattr(target=api_http, name="global_store_port_lock", value=asyncio.Lock())

    result = asyncio.run(main=api_http.http_start_tcp_store_server(tcp_store_port=29501, rank_id=0, world_size=1))

    assert result == {"status": "ok"}
    old.kill.assert_called_once_with()
    old.wait.assert_called_once_with()
    start.assert_called_once_with("host", 29501)
    assert api_http.global_store_port_to_process == {29501: new}
    assert not api_http.global_store_port_to_client_states


def test_tcp_store_propagates_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target=nccl_tcp_store, name="start_parent_check_thread", value=lambda parent_pid: None)
    monkeypatch.setattr(target=nccl_tcp_store, name="get_unique_server_name", value=lambda: "test")
    monkeypatch.setattr(target=dist, name="TCPStore", value=Mock(side_effect=RuntimeError("port already in use")))
    with pytest.raises(expected_exception=RuntimeError, match="port already in use"):
        nccl_tcp_store._start_tcp_store_server(nccl_store_host="127.0.0.1", nccl_store_port=29501, parent_pid=123)


def test_parent_monitor_uses_launch_parent_after_reparenting(monkeypatch: pytest.MonkeyPatch) -> None:
    active = Mock(return_value=False)
    kill_children = Mock()
    kill = Mock(side_effect=SystemExit)
    monkeypatch.setattr(target=process_check, name="is_process_active", value=active)
    monkeypatch.setattr(target=process_check, name="kill_child_processes", value=kill_children)
    monkeypatch.setattr(target=process_check.os, name="kill", value=kill)
    monkeypatch.setattr(target=process_check.os, name="getppid", value=Mock(side_effect=AssertionError("reparented")))
    with pytest.raises(expected_exception=SystemExit):
        process_check.check_parent_alive(parent_pid=123)
    active.assert_called_once_with(123)
    kill_children.assert_called_once_with(os.getpid())
