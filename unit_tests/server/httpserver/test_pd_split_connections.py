import asyncio
import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs import FinishStatus, SamplingParams, StartArgs
from lightllm.server.httpserver import pd_loop
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.httpserver_for_pd_master.manager import PDManager
from lightllm.server.pd_io_struct import ObjType, PD_Master_Obj

pytestmark = pytest.mark.usefixtures("single_http_worker_start_args")


def _registration(args, dp_index=None, mode="prefill"):
    info = {
        "node_id": 1,
        "client_ip_port": "p:8000" if mode == "prefill" else "d:8000",
        "mode": mode,
        "start_args": dict(vars(args), dp=8, nnodes=2),
    }
    if dp_index is not None:
        info["dp_index"] = dp_index
    return info


def test_split_connections_register_select_update_and_reconnect_independently(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", '["p:8000", "d:8000"]')
    args = StartArgs(select_p_d_node_strategy="cache_aware", pd_master_mode="1p1d")
    manager = PDManager(args)
    sockets = [object() for _ in range(4)]
    for rank, websocket in enumerate(sockets):
        manager.register_pd(_registration(args, rank), websocket)
    manager.register_pd(_registration(args, mode="decode"), object())
    assert len(manager.prefill_nodes) == 4
    assert manager.is_pd_nodes_ready()
    assert {node.to_llm_url() for node in manager.prefill_nodes} == {"http://p:8000/pd_generate_stream"}

    prompt = "x" * 1025
    cached_node = manager.prefill_nodes[3]
    asyncio.run(manager.selector.insert_prompt_cache(prompt, cached_node))
    selected, _, extra = asyncio.run(manager.selector.select_p_d_node(prompt, None, None))
    assert selected is cached_node
    assert extra.estimated_cache_hit_rate == 1.0
    for rank in range(4):
        manager.update_node_load_info({"connection_key": f"p:8000/dp{rank}", "total_token_usage_rate": rank / 10})
    assert [node.run_status.total_token_usage_rate for node in manager.prefill_nodes] == [0, 0.1, 0.2, 0.3]
    replacement_socket = object()
    replacement_info = dict(_registration(args, 3), connection_id="replacement")
    manager.register_pd(replacement_info, replacement_socket)
    replacement = manager.url_to_pd_nodes["p:8000/dp3"]
    assert replacement.connection_key == cached_node.connection_key
    manager.remove_pd(_registration(args, 3))
    assert manager.url_to_pd_nodes["p:8000/dp3"] is replacement
    assert replacement in manager.prefill_nodes
    assert replacement in manager.selector.prefill_nodes
    manager.remove_pd(replacement_info)
    assert len(manager.prefill_nodes) == 3
    assert "p:8000/dp3" not in manager.url_to_pd_nodes


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_unsplit_reconnection_ignores_old_connection_id_cleanup(monkeypatch, mode):
    monkeypatch.setenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", '["p:8000", "d:8000"]')
    args = StartArgs()
    manager = PDManager(args)
    info = dict(_registration(args, mode=mode), connection_id="old")
    new_info = dict(info, connection_id="new")
    old_socket, new_socket = object(), object()
    manager.register_pd(info, old_socket)
    manager.register_pd(new_info, new_socket)
    key = f"{info['client_ip_port']}/dpNone"
    replacement = manager.url_to_pd_nodes[key]
    manager.remove_pd(info)
    assert manager.url_to_pd_nodes[key] is replacement
    assert replacement.websocket is new_socket
    manager.remove_pd(new_info)
    assert key not in manager.url_to_pd_nodes


@pytest.mark.parametrize(
    "split, dp, nnodes, expected",
    [(False, 8, 2, [None]), (True, 8, 2, [0, 1, 2, 3]), (True, 1, 1, [0]), (True, 1, 2, [0])],
)
def test_prefill_opens_local_dp_connections(monkeypatch, split, dp, nnodes, expected):
    async def run():
        args = StartArgs(host="p", run_mode="prefill", dp=dp, nnodes=nnodes, use_dp_split_mode_connect_pd_master=split)
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.args = args
        manager.pd_mode = pd_loop.NodeRole.P
        manager.is_multinode_tp_slave = False
        manager.recycle_resource_loop = AsyncMock()
        started = []

        async def connect(_manager, dp_index=None):
            started.append(dp_index)

        async def receive():
            while len(started) < len(expected):
                await asyncio.sleep(0)
            raise asyncio.CancelledError()

        manager.zmq_recv_socket = SimpleNamespace(recv_pyobj=receive)
        timer = AsyncMock()
        monkeypatch.setattr(pd_loop, "timer_log", timer)
        monkeypatch.setattr(pd_loop, "pd_handle_loop", connect)
        with pytest.raises(asyncio.CancelledError):
            await manager.handle_loop()
        assert started == expected
        timer.assert_awaited_once_with(manager)

    asyncio.run(run())


@pytest.mark.parametrize("dp_index", [None, 0, 3])
def test_pd_handle_loop_passes_its_fixed_rank_to_each_master(monkeypatch, dp_index):
    async def run():
        manager = SimpleNamespace(args=StartArgs(host="p", run_mode="prefill"))
        started = []
        tasks = []
        masters = {rank: PD_Master_Obj(rank, f"master{rank}:8000") for rank in (1, 2)}

        async def connect(_manager, master, rank):
            started.append((master.node_id, rank))
            tasks.append(asyncio.current_task())
            await asyncio.Event().wait()

        async def stop_after_connections(_seconds):
            while len(started) < len(masters):
                await original_sleep(0)
            raise asyncio.CancelledError()

        original_sleep = asyncio.sleep
        monkeypatch.setattr(pd_loop, "_get_pd_master_objs", AsyncMock(return_value=masters))
        monkeypatch.setattr(pd_loop, "_pd_handle_task", connect)
        monkeypatch.setattr(pd_loop.asyncio, "sleep", stop_after_connections)
        try:
            with pytest.raises(asyncio.CancelledError):
                await pd_loop.pd_handle_loop(manager, dp_index)
            assert started == [(1, dp_index), (2, dp_index)]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("dp_index", [None, 0, 3])
def test_connection_registers_rank_binds_requests_and_reports_rank_load(monkeypatch, caplog, dp_index):
    from lightllm.server import api_http

    monkeypatch.setenv("LIGHTLLM_PD_REGISTER_TOKEN", "test-pd-secret")
    caplog.set_level("INFO")
    monkeypatch.setattr(pd_loop.logger, "handlers", [caplog.handler])

    async def run():
        args = StartArgs(run_mode="prefill", dp=8, nnodes=2)
        manager = SimpleNamespace(args=args, host_ip="p", pd_mode=SimpleNamespace(value="prefill"))
        monkeypatch.setattr(
            api_http,
            "g_objs",
            SimpleNamespace(
                args=args,
                httpserver_manager=manager,
                shared_token_load=SimpleNamespace(get_dynamic_max_load=lambda rank: rank / 10),
            ),
        )
        monkeypatch.setattr(pd_loop, "get_shm_port_args", lambda: SimpleNamespace(port=8000))
        params = SamplingParams()
        params.init(None, max_new_tokens=1)
        params.group_request_id = 123
        params.suggested_dp_index = -1
        dispatched = asyncio.Event()
        payloads = []

        async def generate(**kwargs):
            req_params = kwargs["sampling_params"]
            assert req_params.suggested_dp_index == (-1 if dp_index is None else dp_index)
            assert kwargs["pd_upload_websocket"] is websocket
            await kwargs["forwarding_queue"].put((123, "P", {}, FinishStatus()))
            dispatched.set()

        requests = 0

        async def receive():
            nonlocal requests
            requests += 1
            if requests == 1:
                return pickle.dumps((ObjType.REQ, ("prompt", params, None)))
            await dispatched.wait()
            while not any(isinstance(payload, tuple) and payload[0] == ObjType.TOKEN_PACKS for payload in payloads):
                await asyncio.sleep(0)
            raise asyncio.CancelledError()

        async def send(payload):
            payloads.append(json.loads(payload) if isinstance(payload, str) else pickle.loads(payload))

        websocket = SimpleNamespace(
            send=send, recv=receive, transport=SimpleNamespace(get_extra_info=lambda _key: MagicMock())
        )

        class Connection:
            async def __aenter__(self):
                return websocket

            async def __aexit__(self, *_args):
                return False

        connections = MagicMock(side_effect=[OSError("retry"), Connection()])
        monkeypatch.setattr(pd_loop.websockets, "connect", connections)
        original_sleep = asyncio.sleep

        async def skip_retry_delay(seconds):
            await original_sleep(0 if seconds == 10 else seconds)

        monkeypatch.setattr(pd_loop.asyncio, "sleep", skip_retry_delay)
        monkeypatch.setattr(pd_loop, "_pd_process_generate", generate)
        await asyncio.wait_for(pd_loop._pd_handle_task(manager, PD_Master_Obj(1, "master:8000"), dp_index), 2)
        assert payloads[0].get("dp_index") == dp_index
        assert "dp_index" in payloads[0]
        assert connections.call_count == 2
        for call in connections.call_args_list:
            assert call.kwargs["additional_headers"] == {"Authorization": "Bearer test-pd-secret"}
        assert "test-pd-secret" not in json.dumps(payloads[0])
        assert "Sent registration JSON" in caplog.text
        assert "test-pd-secret" not in caplog.text
        for payload in payloads[1:]:
            if payload[0] != ObjType.TOKEN_PACKS:
                continue
            load = payload[-1]
            assert "dp_index" not in load
            assert load["connection_key"] == f"p:8000/dp{dp_index}"
            assert load["total_token_usage_rate"] == pytest.approx(0.15 if dp_index is None else dp_index / 10)
        assert args.host != "p"  # Registration must not mutate shared startup arguments.

    asyncio.run(run())


@pytest.mark.parametrize("dp_index", [None, 0, 3])
@pytest.mark.parametrize("with_source", [False, True], ids=["legacy", "connection-source"])
def test_reported_load_key_matches_registered_connection_without_rewriting(monkeypatch, dp_index, with_source):
    from fastapi import WebSocketDisconnect
    from lightllm.server import api_http, api_http_pd

    monkeypatch.setenv("LIGHTLLM_PD_REGISTER_TOKEN", "test-pd-secret")
    monkeypatch.setenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", '["p:8000", "d:8000"]')
    args = StartArgs(dp=8, nnodes=2, pd_node_id=1)
    manager = PDManager(args)
    info = _registration(args, dp_index)
    if with_source:
        info["connection_id"] = f"load-source-{dp_index}"
    monkeypatch.setattr(
        api_http,
        "g_objs",
        SimpleNamespace(
            args=args,
            httpserver_manager=SimpleNamespace(host_ip="p"),
            shared_token_load=SimpleNamespace(get_dynamic_max_load=lambda _rank: 0.7),
        ),
    )
    monkeypatch.setattr(pd_loop, "get_shm_port_args", lambda: SimpleNamespace(port=8000))
    load_info = pd_loop._get_load_info(dp_index)
    message = (ObjType.TOKEN_PACKS, [], load_info)
    messages = iter([pickle.dumps(message)])
    updated_nodes = []

    async def receive():
        try:
            return next(messages)
        except StopIteration:
            raise WebSocketDisconnect()

    async def handle(obj):
        assert obj[2] == load_info
        manager.update_node_load_info(obj[2])
        updated_nodes.append(manager.url_to_pd_nodes[obj[2]["connection_key"]])

    websocket = SimpleNamespace(
        headers={"authorization": "Bearer test-pd-secret"},
        accept=AsyncMock(),
        client=("p", 8000),
        receive_text=AsyncMock(return_value=json.dumps(info)),
        receive_bytes=receive,
    )
    monkeypatch.setattr(
        api_http,
        "g_objs",
        SimpleNamespace(
            httpserver_manager=SimpleNamespace(
                register_pd=AsyncMock(side_effect=manager.register_pd),
                remove_pd=AsyncMock(side_effect=manager.remove_pd),
                put_to_handle_queue=handle,
            )
        ),
    )
    asyncio.run(api_http_pd.register_and_keep_alive(websocket))
    assert len(updated_nodes) == 1
    assert updated_nodes[0].dp_index == dp_index
    assert updated_nodes[0].websocket is websocket
    assert updated_nodes[0].run_status.total_token_usage_rate == 0.7


@pytest.mark.parametrize("node_id, mode, rank", [(2, "prefill", 0), (2, "prefill", 1), (1, "decode", 1)])
def test_conflicting_address_owner_preserves_routing(monkeypatch, node_id, mode, rank):
    monkeypatch.setenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", '["p:8000"]')
    args = StartArgs(select_p_d_node_strategy="cache_aware")
    manager = PDManager(args)
    websocket = object()
    manager.register_pd(_registration(args, 0), websocket)
    incumbent = manager.url_to_pd_nodes["p:8000/dp0"]
    before = (manager.url_to_pd_nodes.copy(), manager.prefill_nodes[:], manager.decode_nodes[:])
    info = dict(_registration(args, rank), node_id=node_id, mode=mode)
    with pytest.raises(ValueError):
        manager.register_pd(info, object())
    assert (manager.url_to_pd_nodes, manager.prefill_nodes, manager.decode_nodes) == before
    assert incumbent.websocket is websocket
    assert manager.selector.prefill_nodes == [incumbent]
    assert manager.selector.decode_nodes == []


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254:80",
        "10.0.0.99:8000",
        "http://p:8000",
        "user@p:8000",
        "p:8000/path",
        "p:8000?query",
        "p:8000#fragment",
        " p:8000",
        "p:8000\n",
        "p",
        "p:0",
        "p:65536",
        "p:-1",
        "p:http",
        "p:08000",
        "P:8000",
        "p.:8000",
        "127.1:8000",
        "0x7f000001:8000",
        "[::1%lo]:8000",
        "::1:8000",
    ],
)
def test_rejected_destination_never_reaches_health_checks(monkeypatch, address):
    enrolled = "p:8000" if address in ("169.254.169.254:80", "10.0.0.99:8000") else address
    monkeypatch.setenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", json.dumps([enrolled]))
    args = StartArgs()
    manager = PDManager(args)
    with pytest.raises(ValueError):
        manager.register_pd(dict(_registration(args), client_ip_port=address), object())
    client = MagicMock()
    monkeypatch.setattr("lightllm.server.httpserver_for_pd_master.manager.httpx.AsyncClient", client)
    assert asyncio.run(manager.check_pd_nodes_health())
    client.assert_not_called()
    assert manager.url_to_pd_nodes == {}
    assert manager.prefill_nodes == manager.decode_nodes == []


@pytest.mark.parametrize("config", [None, "", "[]", "not json", "{}", '["p:8000", null]', '["p:8000/path"]'])
def test_invalid_allowlist_fails_closed(monkeypatch, config):
    if config is None:
        monkeypatch.delenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", raising=False)
    else:
        monkeypatch.setenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", config)
    args = StartArgs()
    manager = PDManager(args)
    with pytest.raises(ValueError):
        manager.register_pd(_registration(args), object())
    assert manager.url_to_pd_nodes == {}


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("address", ["10.0.0.2:8000", "[fd00::2]:8000", "pd-node.cluster.local:8000"])
def test_enrolled_destinations_use_exact_health_urls(monkeypatch, address, mode):
    monkeypatch.setenv("LIGHTLLM_PD_ALLOWED_NODE_ADDRESSES", json.dumps([address]))
    args = StartArgs()
    manager = PDManager(args)
    manager.register_pd(dict(_registration(args, mode=mode), client_ip_port=address), object())
    client = AsyncMock()
    client.get.return_value = SimpleNamespace(status_code=200)
    client.__aenter__.return_value = client
    monkeypatch.setattr("lightllm.server.httpserver_for_pd_master.manager.httpx.AsyncClient", lambda **_kwargs: client)
    assert asyncio.run(manager.check_pd_nodes_health())
    client.get.assert_awaited_once_with(f"http://{address}/health")


def test_split_option_defaults_off_and_is_prefill_only():
    from lightllm.server.api_server import launch_server

    parser = make_argument_parser()
    assert parser.parse_args([]).use_dp_split_mode_connect_pd_master is False
    assert parser.parse_args(["--use_dp_split_mode_connect_pd_master"]).use_dp_split_mode_connect_pd_master is True
    assert StartArgs().use_dp_split_mode_connect_pd_master is False
    with pytest.raises(ValueError, match="only supports prefill"):
        launch_server(StartArgs(run_mode="decode", use_dp_split_mode_connect_pd_master=True))
