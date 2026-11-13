import asyncio
import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from lightllm.server import api_http
from lightllm.server.core.objs import StartArgs
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.server.pd_io_struct import PDUpKVStatus


def _install(monkeypatch, *, ready=False, healthy=False):
    # Exercise the real HTTP/WS handlers without loading a tokenizer/model.
    monkeypatch.setattr(api_http.app.router, "on_startup", [])
    monkeypatch.setattr(api_http.app.router, "on_shutdown", [])
    args = StartArgs(
        run_mode="pd_master",
        host="10.0.0.1",
        port=18090,
        pd_node_id=100,
    )
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.__dict__.update(
        args=args,
        pd_manager=SimpleNamespace(
            prefill_nodes=[SimpleNamespace(node_id=199)],
            decode_nodes=[SimpleNamespace(node_id=200)],
            is_pd_nodes_ready=Mock(return_value=ready),
            check_pd_nodes_health=AsyncMock(return_value=True),
        ),
        is_healthy=Mock(return_value=healthy),
        update_req_status=AsyncMock(),
    )
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(args=args, httpserver_manager=manager))
    return args, manager


@pytest.mark.parametrize("ready,healthy", [(False, False), (True, False), (True, True)])
def test_local_readiness_and_health_preserve_distinct_checks(monkeypatch, ready, healthy):
    _, manager = _install(monkeypatch, ready=ready, healthy=healthy)
    readiness = api_http.readiness()
    assert readiness.status_code == (200 if ready else 503)
    assert json.loads(readiness.body) == {"status": "ok" if ready else "not ready"}
    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == (200 if ready and healthy else 503)
    assert json.loads(response.body) == {
        "inference_healthy": healthy,
        "pd_nodes_ready": ready,
        "message": "Ok" if ready and healthy else "Error",
        "pd_master_mode": "elastic",
        "registered_prefill_nodes": 1,
        "registered_decode_nodes": 1,
    }
    manager.pd_manager.check_pd_nodes_health.assert_not_awaited()


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1)


def test_local_disconnect_makes_readiness_and_health_unavailable(monkeypatch):
    _, manager = _install(monkeypatch, ready=True, healthy=True)
    manager.pd_manager.is_pd_nodes_ready.return_value = False
    readiness = api_http.readiness()
    health = asyncio.run(api_http.healthcheck(None))
    assert readiness.status_code == health.status_code == 503
    assert json.loads(readiness.body) == {"status": "not ready"}
    assert json.loads(health.body)["pd_nodes_ready"] is False
    assert json.loads(health.body)["message"] == "Error"


def test_fixed_topology_health_failure_has_consistent_payload(monkeypatch):
    args, manager = _install(monkeypatch, ready=True, healthy=True)
    args.pd_master_mode = "1p1d"
    manager.pd_manager.check_pd_nodes_health.return_value = False
    response = asyncio.run(api_http.healthcheck(None))
    assert response.status_code == 503
    assert json.loads(response.body)["pd_nodes_healthy"] is False
    assert json.loads(response.body)["message"] == "Error"
    manager.pd_manager.check_pd_nodes_health.assert_awaited_once()


def test_kv_status_keeps_existing_wire_protocol(monkeypatch):
    _, manager = _install(monkeypatch)
    with TestClient(api_http.app) as client:
        with client.websocket_connect("/kv_move_status") as websocket:
            websocket.send_bytes(pickle.dumps(PDUpKVStatus(8, 100, b"metadata")))
            client.portal.call(_wait_until, lambda: manager.update_req_status.await_count == 1)
