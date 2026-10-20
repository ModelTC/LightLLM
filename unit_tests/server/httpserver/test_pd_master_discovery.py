import asyncio
import base64
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from lightllm.server.httpserver import pd_loop
from lightllm.server.pd_io_struct import PD_Master_Obj


def _registry_payload(masters):
    return {"data": base64.b64encode(pickle.dumps(masters)).decode("ascii")}


def _install_http(monkeypatch, handler):
    original_client = httpx.AsyncClient
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(pd_loop.httpx, "AsyncClient", client)
    monkeypatch.setattr(
        pd_loop, "get_shm_port_args", lambda: SimpleNamespace(pd_master_port=8000, config_server_port=9000)
    )
    return options


@pytest.mark.parametrize("config_host,config_port", [(None, None), ("config", None), (None, 9000)])
@pytest.mark.parametrize("worker_ports", [[8000], [8100, 8101]])
def test_without_config_server_reads_worker_ports_before_connecting(
    monkeypatch, config_host, config_port, worker_ports
):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if request.url.path == "/get_server_info":
            worker_port = request.url.port
            return httpx.Response(
                200,
                json={
                    "run_mode": "pd_master",
                    "pd_master_multi_ports": worker_ports,
                    "pd_node_id": 100 + max(0, worker_port - 8100),
                    "host": "internal-master",
                    "port": worker_port,
                    "httpserver_workers": len(worker_ports),
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    options = _install_http(monkeypatch, handler)
    args = SimpleNamespace(config_server_host=config_host, config_server_port=config_port, pd_master_ip="seed")
    masters = asyncio.run(pd_loop._get_pd_master_objs(args))
    if len(worker_ports) > 1:
        assert masters == {
            8100: PD_Master_Obj(8100, "seed:8100"),
            8101: PD_Master_Obj(8101, "seed:8101"),
        }
        assert requests == ["http://seed:8000/get_server_info"]
    else:
        assert masters == {100: PD_Master_Obj(100, "seed:8000")}
        assert requests == ["http://seed:8000/get_server_info"]
    assert options == [{"trust_env": False}]


@pytest.mark.parametrize("status", [404, 405, 500, 503])
def test_mode_probe_http_errors_do_not_guess_a_single_master(monkeypatch, status):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(status)

    _install_http(monkeypatch, handler)
    sleep = AsyncMock()
    monkeypatch.setattr(pd_loop.asyncio, "sleep", sleep)
    args = SimpleNamespace(config_server_host=None, config_server_port=None, pd_master_ip="seed")

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) is None
    assert requests == ["http://seed:8000/get_server_info"]
    sleep.assert_awaited_once_with(10)


@pytest.mark.parametrize(
    "info",
    [
        {"run_mode": "prefill", "pd_master_multi_ports": [8000], "pd_node_id": 0},
        {"run_mode": "pd_master", "pd_node_id": 0},
        {"run_mode": "pd_master", "pd_master_multi_ports": [], "pd_node_id": 0},
    ],
)
def test_invalid_mode_responses_do_not_guess_a_single_master(monkeypatch, info):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, json=info)

    _install_http(monkeypatch, handler)
    monkeypatch.setattr(pd_loop.asyncio, "sleep", AsyncMock())
    args = SimpleNamespace(config_server_host=None, config_server_port=None, pd_master_ip="seed")

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) is None
    assert requests == ["http://seed:8000/get_server_info"]


def test_discovered_multiprocess_master_does_not_query_worker_ports(monkeypatch):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "run_mode": "pd_master",
                "httpserver_workers": 2,
                "pd_master_multi_ports": [8100, 8101],
            },
        )

    _install_http(monkeypatch, handler)
    args = SimpleNamespace(config_server_host=None, config_server_port=None, pd_master_ip="seed")

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) == {
        8100: PD_Master_Obj(8100, "seed:8100"),
        8101: PD_Master_Obj(8101, "seed:8101"),
    }
    assert requests == ["http://seed:8000/get_server_info"]


def test_config_server_discovery_keeps_the_original_registry_protocol_and_uses_http(monkeypatch):
    requests = []
    registered = {100: PD_Master_Obj(100, "master:8100"), 101: PD_Master_Obj(101, "master:8101")}

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, json=_registry_payload(registered))

    options = _install_http(monkeypatch, handler)
    args = SimpleNamespace(config_server_host="config", config_server_port=9000)

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) == registered
    assert requests == ["http://config:9000/registered_objects"]
    assert options == [{"trust_env": False}]


@pytest.mark.parametrize("status", [404, 405, 400, 401, 500, 503])
def test_config_server_http_errors_do_not_fall_back_to_a_direct_master(monkeypatch, status):
    _install_http(monkeypatch, lambda request: httpx.Response(status))
    args = SimpleNamespace(config_server_host="config", config_server_port=9000, pd_master_ip="seed")

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) is None


@pytest.mark.parametrize("failure", ["network", "invalid_pickle", "base_exception"])
def test_discovery_exception_is_throttled_without_falling_back(monkeypatch, failure):
    class DiscoveryFailure(BaseException):
        pass

    def handler(request):
        if failure == "network":
            raise httpx.ConnectError("not ready", request=request)
        if failure == "base_exception":
            raise DiscoveryFailure("discovery failed")
        return httpx.Response(200, json={"data": base64.b64encode(b"invalid-pickle").decode()})

    _install_http(monkeypatch, handler)
    sleep = AsyncMock()
    monkeypatch.setattr(pd_loop.asyncio, "sleep", sleep)
    args = SimpleNamespace(config_server_host="config", config_server_port=9000, pd_master_ip="seed")

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) is None
    sleep.assert_awaited_once_with(10)


def test_discovery_follows_config_registry_registration_and_removal(monkeypatch):
    registered = {}
    _install_http(monkeypatch, lambda request: httpx.Response(200, json=_registry_payload(registered)))
    args = SimpleNamespace(config_server_host="config", config_server_port=9000)

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) == {}
    registered[100] = PD_Master_Obj(100, "master:8100")
    registered[101] = PD_Master_Obj(101, "[2001:db8::1]:8101")
    assert asyncio.run(pd_loop._get_pd_master_objs(args)) == registered
    del registered[100]
    assert asyncio.run(pd_loop._get_pd_master_objs(args)) == registered
    registered.clear()
    assert asyncio.run(pd_loop._get_pd_master_objs(args)) == {}


@pytest.mark.parametrize("host", ["::1", "[::1]", "2001:db8::1234"])
def test_ipv6_direct_master_address_is_bracketed(monkeypatch, host):
    _install_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "run_mode": "pd_master",
                "httpserver_workers": 1,
                "pd_master_multi_ports": [8000],
                "pd_node_id": 0,
                "port": 8000,
            },
        ),
    )
    args = SimpleNamespace(config_server_host=None, config_server_port=None, pd_master_ip=host)
    masters = asyncio.run(pd_loop._get_pd_master_objs(args))
    expected_host = host if host.startswith("[") else f"[{host}]"

    assert masters == {0: PD_Master_Obj(0, f"{expected_host}:8000")}


def test_ipv6_config_server_address_is_bracketed_for_http(monkeypatch):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(200, json=_registry_payload({}))

    _install_http(monkeypatch, handler)
    args = SimpleNamespace(config_server_host="2001:db8::1", config_server_port=9000)

    assert asyncio.run(pd_loop._get_pd_master_objs(args)) == {}
    assert requests == ["http://[2001:db8::1]:9000/registered_objects"]


def test_discovery_cancellation_closes_http_client_and_uses_failure_retry(monkeypatch):
    async def run():
        started = asyncio.Event()
        closed = asyncio.Event()

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                closed.set()

            async def get(self, _url):
                started.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(pd_loop.httpx, "AsyncClient", lambda **_kwargs: Client())
        monkeypatch.setattr(pd_loop, "get_shm_port_args", lambda: SimpleNamespace(config_server_port=9000))
        sleep = AsyncMock()
        monkeypatch.setattr(pd_loop.asyncio, "sleep", sleep)
        args = SimpleNamespace(config_server_host="config", config_server_port=9000)
        task = asyncio.create_task(pd_loop._get_pd_master_objs(args))
        await started.wait()
        task.cancel()
        assert await task is None
        sleep.assert_awaited_once_with(10)
        assert closed.is_set()

    asyncio.run(run())


def test_registry_changes_connect_all_owners_and_retire_removed_ids(monkeypatch):
    async def run():
        manager = SimpleNamespace(args=SimpleNamespace(host="prefill"))
        responses = asyncio.Queue()
        discoveries = asyncio.Queue()
        started = asyncio.Queue()
        children = []
        live = set()
        original_sleep = asyncio.sleep

        async def skip_poll_interval(seconds):
            assert seconds == 30, "Topology refresh unexpectedly failed"
            await original_sleep(0)

        async def discover(_args):
            value = await responses.get()
            discoveries.put_nowait(value)
            return value

        async def connect(_manager, master, _rank):
            children.append(asyncio.current_task())
            live.add(master.node_id)
            started.put_nowait(master.node_id)
            try:
                await asyncio.Event().wait()
            finally:
                live.remove(master.node_id)

        monkeypatch.setattr(pd_loop.asyncio, "sleep", skip_poll_interval)
        monkeypatch.setattr(pd_loop, "_get_pd_master_objs", discover)
        monkeypatch.setattr(pd_loop, "_pd_handle_task", connect)
        task = asyncio.create_task(pd_loop.pd_handle_loop(manager))
        try:
            responses.put_nowait({node_id: PD_Master_Obj(node_id, f"master:{node_id}") for node_id in (100, 101)})
            assert {await asyncio.wait_for(started.get(), timeout=1) for _ in range(2)} == {100, 101}
            await discoveries.get()
            responses.put_nowait(None)
            assert await discoveries.get() is None
            await original_sleep(0)
            assert live == {100, 101}
            responses.put_nowait({node_id: PD_Master_Obj(node_id, f"master:{node_id}") for node_id in (101, 200)})
            assert await asyncio.wait_for(started.get(), timeout=1) == 200
            assert live == {101, 200}
            await discoveries.get()
            responses.put_nowait({})
            assert await discoveries.get() == {}
            await original_sleep(0)
            assert not live
        finally:
            task.cancel()
            for child in children:
                child.cancel()
            await asyncio.gather(task, *children, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("dp_index,load", [(None, 0.5), (0, 0.2), (1, 0.8)])
def test_load_report_keeps_its_original_fields_and_formats_ipv6_keys(monkeypatch, dp_index, load):
    from lightllm.server import api_http

    monkeypatch.setattr(
        api_http,
        "g_objs",
        SimpleNamespace(
            args=SimpleNamespace(dp=2, nnodes=1),
            httpserver_manager=SimpleNamespace(host_ip="2001:db8::1"),
            shared_token_load=SimpleNamespace(get_dynamic_max_load=lambda rank: [0.2, 0.8][rank]),
        ),
    )
    monkeypatch.setattr(pd_loop, "get_shm_port_args", lambda: SimpleNamespace(port=8100))

    assert pd_loop._get_load_info(dp_index) == {
        "total_token_usage_rate": load,
        "connection_key": f"[2001:db8::1]:8100/dp{dp_index}",
    }
