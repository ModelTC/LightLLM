"""Existing PD WebSocket registration and message-forwarding behavior."""

import asyncio
import json
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server import api_http, api_http_pd


def _websocket(**overrides):
    values = dict(
        accept=AsyncMock(),
        close=AsyncMock(),
        headers={"authorization": "Bearer test-pd-secret"},
        client=("pd", 18000),
        receive_text=AsyncMock(return_value=json.dumps({"node_id": 1, "mode": "prefill"})),
        receive_bytes=AsyncMock(),
        send_bytes=AsyncMock(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _manager(monkeypatch, **overrides):
    monkeypatch.setenv("LIGHTLLM_PD_REGISTER_TOKEN", "test-pd-secret")
    values = dict(
        register_pd=AsyncMock(),
        remove_pd=AsyncMock(),
        put_to_handle_queue=AsyncMock(),
        args=SimpleNamespace(pd_node_id=100),
        update_req_status=AsyncMock(),
    )
    values.update(overrides)
    manager = SimpleNamespace(**values)
    monkeypatch.setattr(api_http, "g_objs", SimpleNamespace(httpserver_manager=manager))
    return manager


@pytest.mark.parametrize("payload", [None, "{invalid json"])
def test_rejected_registration_does_not_remove_an_existing_connection(monkeypatch, payload):
    manager = _manager(monkeypatch, register_pd=AsyncMock(side_effect=ValueError("wrong model parameters")))
    websocket = _websocket()
    if payload is not None:
        websocket.receive_text.return_value = payload
    asyncio.run(api_http_pd.register_and_keep_alive(websocket))
    websocket.close.assert_awaited_once()
    manager.remove_pd.assert_not_awaited()


@pytest.mark.parametrize(
    "token, header",
    [
        ("test-pd-secret", None),
        ("test-pd-secret", "Bearer wrong"),
        (None, "Bearer test-pd-secret"),
        ("", "Bearer "),
        ("   ", "Bearer    "),
        ("caf\u00e9", "Bearer caf\u00e9"),
    ],
)
def test_registration_authentication_precedes_accept_and_frames(monkeypatch, token, header):
    manager = _manager(monkeypatch)
    if token is None:
        monkeypatch.delenv("LIGHTLLM_PD_REGISTER_TOKEN")
    else:
        monkeypatch.setenv("LIGHTLLM_PD_REGISTER_TOKEN", token)
    websocket = _websocket(headers={} if header is None else {"authorization": header})
    loads = AsyncMock()
    monkeypatch.setattr(api_http_pd.pickle, "loads", loads)
    asyncio.run(api_http_pd.register_and_keep_alive(websocket))
    websocket.close.assert_awaited_once_with(code=1008, reason="PD registration unauthorized")
    websocket.accept.assert_not_awaited()
    websocket.receive_text.assert_not_awaited()
    websocket.receive_bytes.assert_not_awaited()
    manager.register_pd.assert_not_awaited()
    manager.remove_pd.assert_not_awaited()
    manager.put_to_handle_queue.assert_not_awaited()
    loads.assert_not_called()


def test_heartbeat_wait_is_bounded(monkeypatch):
    async def run():
        original_wait_for = asyncio.wait_for
        timeouts = []

        async def wait_forever():
            await asyncio.Event().wait()

        async def short_heartbeat_wait(awaitable, timeout):
            timeouts.append(timeout)
            return await original_wait_for(awaitable, timeout=0.01)

        manager = _manager(monkeypatch)
        monkeypatch.setattr(api_http_pd.asyncio, "wait_for", short_heartbeat_wait)
        websocket = _websocket(receive_bytes=wait_forever)
        await original_wait_for(api_http_pd.register_and_keep_alive(websocket), timeout=1)
        websocket.accept.assert_awaited_once()
        manager.register_pd.assert_awaited_once()
        assert timeouts == [30]
        assert websocket.close.await_args.kwargs["code"] == 1011
        manager.remove_pd.assert_awaited_once()

    asyncio.run(run())


def test_kv_status_is_forwarded_without_registration_or_reply(monkeypatch):
    from fastapi import WebSocketDisconnect
    from lightllm.server.pd_io_struct import PDUpKVStatus

    status = PDUpKVStatus(8, 100, b"metadata")
    manager = _manager(monkeypatch)
    websocket = _websocket(receive_bytes=AsyncMock(side_effect=[pickle.dumps(status), WebSocketDisconnect()]))
    asyncio.run(api_http_pd.kv_move_status(websocket))
    manager.update_req_status.assert_awaited_once_with(status)
    websocket.send_bytes.assert_not_awaited()
