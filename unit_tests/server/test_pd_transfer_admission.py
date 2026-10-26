import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import WebSocketDisconnect

from lightllm.server import api_http, api_http_pd
from lightllm.server.httpserver import pd_loop
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, PDManager
from lightllm.server.httpserver_for_pd_master.pd_selector import pd_node_stats_client, prompt_cache_tree_client
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj, PDTransferStatus
from lightllm.utils.error_utils import PDTransportUnavailableError


def _status():
    status = PDTransferStatus.__new__(PDTransferStatus)
    status.page_capacity = 2
    status.quarantined_pages = SimpleNamespace(arr=np.zeros(8, dtype=np.int64))
    return status


def test_partial_quarantine_retains_capacity_and_safe_release_restores_node():
    status = _status()
    status.set_quarantined_pages(5, 1)
    assert status.is_available()
    status.set_quarantined_pages(5, 2)
    assert not status.is_available()
    status.set_quarantined_pages(5, 1)
    assert status.is_available()


@pytest.fixture
def managers(monkeypatch):
    args = SimpleNamespace(
        httpserver_workers=1,
        select_p_d_node_strategy="random",
        pd_master_mode="2p2d",
        config_server_host=None,
        config_server_port=None,
    )
    monkeypatch.setattr(pd_node_stats_client, "get_env_start_args", lambda: args)
    monkeypatch.setattr(prompt_cache_tree_client, "get_env_start_args", lambda: args)

    def create(strategy="random"):
        args.select_p_d_node_strategy = strategy
        manager = PDManager(args)
        manager.prefill_nodes = [PD_Client_Obj(i, f"p{i}:8000", "prefill", {}) for i in range(2)]
        manager.decode_nodes = [PD_Client_Obj(i + 2, f"d{i}:8000", "decode", {}) for i in range(2)]
        manager.url_to_pd_nodes = {n.connection_key: n for n in manager.prefill_nodes + manager.decode_nodes}
        manager.selector.update_nodes(manager.prefill_nodes, manager.decode_nodes)
        return manager

    return create


@pytest.mark.parametrize("strategy", ["random", "round_robin", "adaptive_load", "cache_aware"])
def test_all_selectors_exclude_unavailable_nodes_and_restore_without_unregistering(managers, strategy):
    manager = managers(strategy)
    p0, p1 = manager.prefill_nodes
    d0, d1 = manager.decode_nodes
    manager.update_transfer_availability(p0, False)
    manager.update_transfer_availability(d0, False)
    p, d, _ = asyncio.run(manager.select_p_d_node("prompt" * 512, None, None))
    assert p is p1 and d is d1
    assert manager.is_pd_nodes_ready()
    manager.update_transfer_availability(p1, False)
    with pytest.raises(PDTransportUnavailableError) as error:
        asyncio.run(manager.select_p_d_node("prompt" * 512, None, None))
    assert error.value.status_code == 429
    manager.update_transfer_availability(p0, True)
    p, d, _ = asyncio.run(manager.select_p_d_node("prompt" * 512, None, None))
    assert p is p0 and d is d1
    assert manager.is_pd_nodes_ready()


def test_delayed_heartbeat_cannot_disable_replacement_connection(managers):
    manager = managers()
    old = manager.prefill_nodes[0]
    replacement = PD_Client_Obj(old.node_id, old.client_ip_port, old.mode, {})
    manager.prefill_nodes[0] = replacement
    manager.url_to_pd_nodes[replacement.connection_key] = replacement
    manager.selector.update_nodes(manager.prefill_nodes, manager.decode_nodes)
    manager.update_transfer_availability(old, False)
    assert replacement.run_status.transfer_available
    assert replacement in manager.selector.prefill_nodes


def test_decode_becomes_unavailable_during_async_cache_lookup_returns_429(managers):
    manager = managers("cache_aware")

    async def lookup(*args, **kwargs):
        for node in manager.decode_nodes:
            manager.update_transfer_availability(node, False)
        return manager.prefill_nodes[0], None

    manager.selector.policy.select_worker = lookup
    with pytest.raises(PDTransportUnavailableError):
        asyncio.run(manager.select_p_d_node("prompt" * 512, None, None))


def test_admission_rejects_before_grammar_or_request_allocation():
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.transfer_status = _status()
    manager.transfer_status.set_quarantined_pages(0, 2)
    params = SimpleNamespace(get_compiled_grammar=AsyncMock())

    async def run():
        async for _ in manager._generate("prompt", params, None, None):
            pytest.fail("unavailable node must not generate")

    with pytest.raises(PDTransportUnavailableError):
        asyncio.run(run())
    params.get_compiled_grammar.assert_not_awaited()


def test_master_does_not_busy_retry_when_no_safe_candidates_remain():
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    attempts = []

    async def attempt(*args):
        attempts.append(1)
        raise PDTransportUnavailableError()
        yield

    manager._generate_one_attempt = attempt
    manager.enable_pd_node_self_request_limit = True

    async def run():
        async for _ in manager._generate_one("prompt", None, None, None, 0, 1, 1):
            pytest.fail("no token expected")

    with pytest.raises(PDTransportUnavailableError):
        asyncio.run(run())
    assert attempts == [1]


def test_heartbeat_reports_fault_and_recovery_without_waiting_15_seconds(monkeypatch):
    status = _status()
    messages = []
    polls = []

    async def send(data):
        messages.append(pickle.loads(data))

    async def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 1:
            status.set_quarantined_pages(0, 2)
        elif len(polls) == 3:
            status.set_quarantined_pages(0, 0)
        elif len(polls) == 4:
            raise asyncio.CancelledError()

    monkeypatch.setattr(pd_loop.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pd_loop._PDHeartbeatReporter(SimpleNamespace(send=send), status).run())
    assert messages == [(ObjType.HEARTBEAT, True), (ObjType.HEARTBEAT, False), (ObjType.HEARTBEAT, True)]
    assert polls == [1, 1, 1, 1]


def test_healthy_heartbeat_keeps_existing_15_second_network_interval(monkeypatch):
    messages = []
    clock = SimpleNamespace(now=0)

    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "time", lambda: clock.now)

        async def sleep(seconds):
            clock.now += seconds
            if clock.now == 31:
                raise asyncio.CancelledError()

        async def send(data):
            messages.append((clock.now, pickle.loads(data)))

        monkeypatch.setattr(pd_loop.asyncio, "sleep", sleep)
        with pytest.raises(asyncio.CancelledError):
            await pd_loop._PDHeartbeatReporter(SimpleNamespace(send=send), None).run()

    asyncio.run(run())
    assert messages == [(t, (ObjType.HEARTBEAT, True)) for t in (0, 15, 30)]


def test_local_admission_reports_unavailable_before_busy_reply():
    async def generate(**kwargs):
        raise PDTransportUnavailableError()
        yield

    manager = SimpleNamespace(generate=generate, cancel_pd_request_registration=lambda _: None)
    socket = SimpleNamespace(send=AsyncMock())
    params = SimpleNamespace(group_request_id=123)
    reporter = pd_loop._PDHeartbeatReporter(socket, None)
    asyncio.run(pd_loop._pd_process_generate(manager, "prompt", params, {}, None, socket, None, reporter))
    messages = [pickle.loads(call.args[0]) for call in socket.send.await_args_list]
    assert messages[0] == (ObjType.HEARTBEAT, False)
    assert messages[1][0:2] == (ObjType.PD_UPLOAD_SERVER_BUSY, 123)


def test_immediate_rejection_and_recovery_between_heartbeat_checks(monkeypatch, managers):
    master = managers()
    node = master.prefill_nodes[0]
    status = _status()
    clock = SimpleNamespace(now=0.0, polls=0)
    messages = []

    async def generate(**kwargs):
        raise PDTransportUnavailableError()
        yield

    manager = SimpleNamespace(generate=generate, cancel_pd_request_registration=lambda _: None)

    async def send(data):
        message = pickle.loads(data)
        if message[0] == ObjType.HEARTBEAT:
            master.update_transfer_availability(node, message[1])
            messages.append((clock.now, message[1]))

    socket = SimpleNamespace(send=send)
    reporter = pd_loop._PDHeartbeatReporter(socket, status)

    async def sleep(seconds):
        clock.polls += 1
        if clock.polls == 2:
            raise asyncio.CancelledError()
        clock.now = 0.1
        status.set_quarantined_pages(0, 2)
        await pd_loop._pd_process_generate(
            manager, "prompt", SimpleNamespace(group_request_id=123), {}, None, socket, None, reporter
        )
        assert node not in master.selector.prefill_nodes
        clock.now = 0.2
        status.set_quarantined_pages(0, 0)
        clock.now = 1.0

    async def run():
        monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: clock.now)
        monkeypatch.setattr(pd_loop.asyncio, "sleep", sleep)
        with pytest.raises(asyncio.CancelledError):
            await reporter.run()

    asyncio.run(run())
    assert messages == [(0.0, True), (0.1, False), (1.0, True)]
    assert node in master.selector.prefill_nodes


def test_heartbeat_cache_follows_serialized_send_completion():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        messages = []

        async def send(data):
            if not messages:
                entered.set()
                await release.wait()
            messages.append(pickle.loads(data))

        reporter = pd_loop._PDHeartbeatReporter(SimpleNamespace(send=send), None)
        periodic = asyncio.create_task(reporter.report())
        await entered.wait()
        immediate = asyncio.create_task(reporter.report(False))
        release.set()
        await asyncio.gather(periodic, immediate)
        assert reporter.last_available is False
        await reporter.report()
        assert messages == [(ObjType.HEARTBEAT, True), (ObjType.HEARTBEAT, False), (ObjType.HEARTBEAT, True)]

    asyncio.run(run())


def test_each_master_applies_heartbeat_to_its_own_registered_connection(monkeypatch, managers):
    for manager in (managers(), managers()):
        node = manager.prefill_nodes[0]
        http_manager = SimpleNamespace(
            register_pd=AsyncMock(return_value=node),
            remove_pd=AsyncMock(),
            put_to_handle_queue=AsyncMock(),
            pd_manager=manager,
        )
        socket = SimpleNamespace(
            client=("127.0.0.1", 12345),
            accept=AsyncMock(),
            receive_text=AsyncMock(return_value="{}"),
            receive_bytes=AsyncMock(side_effect=[pickle.dumps((ObjType.HEARTBEAT, False)), WebSocketDisconnect()]),
        )
        monkeypatch.setattr(api_http.g_objs, "httpserver_manager", http_manager, raising=False)
        asyncio.run(api_http_pd.register_and_keep_alive(socket))
        assert not node.run_status.transfer_available
        assert node not in manager.selector.prefill_nodes
        http_manager.put_to_handle_queue.assert_not_awaited()
