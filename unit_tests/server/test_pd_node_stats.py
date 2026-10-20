"""PD node dispatch statistics stay independent from prompt-cache state."""

import asyncio
import errno
import socket
import threading
import uuid
from unittest.mock import Mock, patch

import pytest
from rpyc.utils.server import ThreadedServer

from lightllm.server.httpserver_for_pd_master.pd_selector import pd_node_stats_client
from lightllm.server.httpserver_for_pd_master.pd_selector import pd_node_stats_server
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_node_stats_client import (
    LocalPDNodeStats,
    RemotePDNodeStats,
    create_pd_node_stats_client,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_node_stats_server import PDNodeStatsState
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.net_utils import get_abstract_unix_socket_address


def _node(address="p", node_id=1):
    return PD_Client_Obj(
        node_id=node_id,
        client_ip_port=address,
        mode="prefill",
        start_args={},
        dp_index=0,
    )


def _counts(node):
    return node.dispatched_prompt_chars, node.dispatched_req_num


def _query(state, *keys):
    return state.get_dispatched_info(list(keys))


def _remote_client(ipc_name="test-pd-node-stats"):
    with patch.object(
        pd_node_stats_client,
        "get_pd_master_unix_socket_address",
        return_value=get_abstract_unix_socket_address("pd_node_stats", ipc_name),
    ):
        return RemotePDNodeStats()


def test_ipc_address_uses_the_launcher_ipc_name(monkeypatch):
    monkeypatch.setattr(pd_node_stats_client, "unix_connect", Mock())
    client = _remote_client("deployment-a")
    assert client.socket_path == get_abstract_unix_socket_address(
        "pd_node_stats", "deployment-a"
    )
    assert client.socket_path != get_abstract_unix_socket_address(
        "pd_node_stats", "deployment-b"
    )


def test_local_client_updates_node_objects_without_rpc(monkeypatch):
    monkeypatch.setattr(
        pd_node_stats_client,
        "get_env_start_args",
        lambda: Mock(httpserver_workers=1),
    )
    client = create_pd_node_stats_client()
    assert isinstance(client, LocalPDNodeStats)
    node = _node()

    async def run():
        await client.add_dispatched_info(p_node=node, chars=10, req_num=2)
        assert _counts(node) == (10, 2)
        await client.sub_dispatched_info(p_node=node, chars=3, req_num=1)
        assert _counts(node) == (7, 1)
        await client.sub_dispatched_info(p_node=node, chars=100, req_num=100)
        assert _counts(node) == (0, 0)
        assert await client.update_dispatched_info([node]) is None

    asyncio.run(run())


def test_remote_client_sends_only_connection_keys(monkeypatch):
    monkeypatch.setattr(pd_node_stats_client, "unix_connect", Mock())
    client = _remote_client()
    client._add_dispatched_info = Mock()
    client._sub_dispatched_info = Mock()
    client._get_dispatched_info = Mock(return_value={"p/dp0": (17, 1)})
    node = _node()

    async def run():
        await client.add_dispatched_info(p_node=node, chars=17, req_num=1)
        await client.sub_dispatched_info(p_node=node, chars=3, req_num=1)
        await client.update_dispatched_info([node])

    asyncio.run(run())
    client._add_dispatched_info.assert_called_once_with(node.connection_key, 17, 1)
    client._sub_dispatched_info.assert_called_once_with(node.connection_key, 3, 1)
    client._get_dispatched_info.assert_called_once_with([node.connection_key])
    assert _counts(node) == (17, 1)


def test_state_creates_keys_lazily_and_clamps_subtraction():
    state = PDNodeStatsState()
    assert _query(state, "a", "b") == {"a": (0, 0), "b": (0, 0)}
    state.add_dispatched_info("a", 10, 2)
    state.sub_dispatched_info("a", 100, 1)
    assert _query(state, "a")["a"] == (0, 1)
    state.sub_dispatched_info("a", 0, 100)
    assert _query(state, "a")["a"] == (0, 0)


@pytest.mark.parametrize("key", ["", None, 1])
def test_invalid_key_does_not_change_statistics(key):
    state = PDNodeStatsState()
    with pytest.raises(ValueError, match="nonempty string"):
        state.add_dispatched_info(key, 1, 1)
    assert not state.stats


def test_idle_and_capacity_eviction_are_lru(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(pd_node_stats_server.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pd_node_stats_server, "MAX_NODE_STATS", 2)
    state = PDNodeStatsState()
    _query(state, "a", "b")
    _query(state, "a")
    _query(state, "c")
    assert list(state.stats) == ["a", "c"]
    now[0] += pd_node_stats_server.NODE_STATS_IDLE_SECONDS
    _query(state, "new")
    assert list(state.stats) == ["new"]


def test_cancelled_update_finishes_before_following_update(monkeypatch):
    monkeypatch.setattr(pd_node_stats_client, "unix_connect", Mock())
    state = PDNodeStatsState()
    client = _remote_client()
    node = _node()
    started, finish = threading.Event(), threading.Event()
    operations = []

    def add_dispatched_info(*args):
        started.set()
        assert finish.wait(5)
        state.add_dispatched_info(*args)
        operations.append("add_dispatched_info")

    def sub_dispatched_info(*args):
        state.sub_dispatched_info(*args)
        operations.append("sub_dispatched_info")

    client._add_dispatched_info = add_dispatched_info
    client._sub_dispatched_info = sub_dispatched_info

    async def run():
        first = asyncio.create_task(
            client.add_dispatched_info(p_node=node, chars=10, req_num=1)
        )
        assert await asyncio.to_thread(started.wait, 5)
        second = asyncio.create_task(
            client.sub_dispatched_info(p_node=node, chars=10, req_num=1)
        )
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second

    asyncio.run(run())
    assert operations == ["add_dispatched_info", "sub_dispatched_info"]
    assert _query(state, node.connection_key)[node.connection_key] == (0, 0)


@pytest.fixture
def rpc_stats_cluster():
    state = PDNodeStatsState()
    ipc_name = uuid.uuid4().hex
    server = ThreadedServer(
        state,
        socket_path=get_abstract_unix_socket_address("pd_node_stats", ipc_name),
        protocol_config={"allow_pickle": True},
    )
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    clients = [_remote_client(ipc_name) for _ in range(4)]
    try:
        yield state, clients
    finally:
        for client in clients:
            connection = client.connection
            if connection is not None:
                connection.close()
        server.close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_real_rpc_clients_share_statistics(rpc_stats_cluster):
    _, clients = rpc_stats_cluster
    node = _node()
    node.websocket = threading.Lock()

    async def run():
        await clients[0].add_dispatched_info(p_node=node, chars=17, req_num=1)
        await clients[1].update_dispatched_info([node])
        assert _counts(node) == (17, 1)
        transport = clients[0].connection._channel.stream.sock
        assert transport.family == socket.AF_UNIX
        await clients[1].sub_dispatched_info(p_node=node, chars=17, req_num=1)
        await clients[0].update_dispatched_info([node])
        assert _counts(node) == (0, 0)

    asyncio.run(run())


def test_abstract_socket_is_released_after_server_close():
    address = get_abstract_unix_socket_address("pd_node_stats", uuid.uuid4().hex)
    server = ThreadedServer(
        PDNodeStatsState(),
        socket_path=address,
        protocol_config={"allow_pickle": True},
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as duplicate:
            with pytest.raises(OSError) as conflict:
                duplicate.bind(address)
            assert conflict.value.errno == errno.EADDRINUSE
    finally:
        server.close()
    ThreadedServer(
        PDNodeStatsState(),
        socket_path=address,
        protocol_config={"allow_pickle": True},
    ).close()


def test_initial_connection_failure_is_reported_during_construction(monkeypatch):
    attempts = []

    def connect(*args, **kwargs):
        attempts.append(1)
        raise ConnectionRefusedError("not listening")

    monkeypatch.setattr(pd_node_stats_client, "unix_connect", connect)

    for _ in range(2):
        with pytest.raises(ConnectionRefusedError):
            _remote_client()

    assert len(attempts) == 2


def test_real_rpc_combined_updates_are_atomic(rpc_stats_cluster):
    state, clients = rpc_stats_cluster
    node = _node()

    async def run():
        await asyncio.gather(
            *(
                clients[index % len(clients)].add_dispatched_info(
                    p_node=node, chars=1, req_num=1
                )
                for index in range(200)
            )
        )

    asyncio.run(run())
    assert _query(state, node.connection_key)[node.connection_key] == (200, 200)
