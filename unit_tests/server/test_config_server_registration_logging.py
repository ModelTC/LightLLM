import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import WebSocketDisconnect

from lightllm.server.config_server import api_http


@pytest.mark.parametrize(
    "endpoint, registry_name",
    [
        ("websocket_endpoint", "registered_pd_master_objs"),
        ("visual_websocket_endpoint", "registered_visual_server_objs"),
    ],
)
@pytest.mark.parametrize("disconnect", [False, True])
def test_registration_logs_and_unregisters_on_exit(monkeypatch, endpoint, registry_name, disconnect):
    other = object()
    registry = {"other": other}
    monkeypatch.setattr(api_http, registry_name, registry)
    logger = Mock()
    monkeypatch.setattr(api_http, "logger", logger)
    error = WebSocketDisconnect() if disconnect else RuntimeError("receive failed")

    async def receive_text():
        assert registry["node"].node_id == "node"
        raise error

    websocket = SimpleNamespace(
        client=("127.0.0.1", 1234),
        accept=AsyncMock(),
        receive_bytes=AsyncMock(return_value=pickle.dumps(SimpleNamespace(node_id="node"))),
        receive_text=receive_text,
    )
    asyncio.run(getattr(api_http, endpoint)(websocket))
    assert registry == {"other": other}
    websocket.accept.assert_awaited_once()
    logger.error.assert_not_called()
    if disconnect:
        logger.exception.assert_not_called()
        assert any(call.args[0] == "Registration connection closed: peer=%s:%s" for call in logger.info.call_args_list)
    else:
        logger.exception.assert_called_once()
        assert "receive failed" in logger.exception.call_args.args[0]
    assert "removed" in logger.info.call_args.args[0]
