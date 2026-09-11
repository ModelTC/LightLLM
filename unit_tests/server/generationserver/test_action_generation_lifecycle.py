import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightllm.server.generationserver.action import server as server_module


def test_partial_tp_startup_closes_workers_that_already_started(monkeypatch):
    first_rpc = SimpleNamespace(close=AsyncMock())
    calls = 0

    async def start_worker():
        nonlocal calls
        calls += 1
        if calls == 1:
            return first_rpc
        raise RuntimeError("rank 1 failed")

    monkeypatch.setattr(server_module, "start_model_process", start_worker)
    manager = server_module.ActionGenerationServer.__new__(
        server_module.ActionGenerationServer
    )
    manager.tp_size = 2
    manager.model_rpcs = []
    manager.args = SimpleNamespace()
    manager.generation_nccl_port = 12345

    with pytest.raises(RuntimeError, match="rank 1 failed"):
        asyncio.run(manager.wait_to_model_ready())

    first_rpc.close.assert_awaited_once_with()
    assert manager.model_rpcs == []


def test_action_manager_close_is_idempotent_and_closes_all_resources():
    manager = server_module.ActionGenerationServer.__new__(
        server_module.ActionGenerationServer
    )
    manager._closed = False
    rpc_one = SimpleNamespace(close=AsyncMock())
    rpc_two = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("close failed")))
    manager.model_rpcs = [rpc_one, rpc_two]
    manager._submissions = {}
    manager._relay_tasks = set()
    manager.transfer = SimpleNamespace(release_all=MagicMock())
    manager.recv_socket = MagicMock()
    manager.context = MagicMock()

    async def scenario():
        await manager.close()
        await manager.close()

    asyncio.run(scenario())

    rpc_one.close.assert_awaited_once_with()
    rpc_two.close.assert_awaited_once_with()
    assert manager.model_rpcs == []
    manager.recv_socket.close.assert_called_once_with(linger=0)
    manager.context.term.assert_called_once_with()


def test_action_manager_run_closes_resources_when_a_loop_fails():
    manager = server_module.ActionGenerationServer.__new__(
        server_module.ActionGenerationServer
    )
    manager.close = AsyncMock()

    async def fail():
        raise RuntimeError("request loop failed")

    manager.loop_for_requests = fail

    with pytest.raises(RuntimeError, match="request loop failed"):
        asyncio.run(manager.run())

    manager.close.assert_awaited_once_with()
