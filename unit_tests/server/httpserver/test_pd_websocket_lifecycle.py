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
        client=("pd", 18000),
        receive_text=AsyncMock(return_value=json.dumps({"node_id": 1, "mode": "prefill"})),
        receive_bytes=AsyncMock(),
        send_bytes=AsyncMock(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _manager(monkeypatch, **overrides):
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


def test_rejected_registration_does_not_remove_an_existing_connection(monkeypatch):
    manager = _manager(monkeypatch, register_pd=AsyncMock(side_effect=ValueError("wrong model parameters")))
    websocket = _websocket()
    with pytest.raises(ValueError, match="wrong model parameters"):
        asyncio.run(api_http_pd.register_and_keep_alive(websocket))
    manager.remove_pd.assert_not_awaited()


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
