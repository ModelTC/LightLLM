import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.utils.error_utils import ServerBusyError


def _manager(*, timeout_seconds: int = 1) -> HttpServerManager:
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.args = SimpleNamespace(run_mode="decode")
    manager.pd_node_request_limit_enabled = True
    manager.pd_node_shm_req_alloc_timeout_seconds = timeout_seconds
    manager.shm_req_manager = MagicMock()
    manager.shm_req_manager.async_release_req_index = AsyncMock()
    return manager


def test_pd_node_regular_request_times_out_when_shm_req_is_unavailable():
    async def run():
        manager = _manager(timeout_seconds=0)
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(return_value=None)

        with pytest.raises(ServerBusyError):
            await manager._alloc_shm_req_indexes(
                1,
                bypass_pd_node_request_limit=False,
            )

    asyncio.run(run())


def test_pd_split_continuation_bypasses_shm_req_allocation_timeout():
    async def run():
        manager = _manager(timeout_seconds=0)
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(side_effect=[None, 7])

        with patch("lightllm.server.httpserver.manager.asyncio.sleep", new=AsyncMock()):
            indexes = await manager._alloc_shm_req_indexes(
                1,
                bypass_pd_node_request_limit=True,
            )

        assert indexes == [7]
        manager.shm_req_manager.async_release_req_index.assert_not_awaited()

    asyncio.run(run())
