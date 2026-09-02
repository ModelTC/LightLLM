import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.router.req_queue.base_queue import BaseQueue


class FakeSharedInt:
    def __init__(self, value: int = 0):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value: int):
        self.value = value


def _manager() -> HttpServerManager:
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.args = SimpleNamespace(run_mode="decode", running_max_req_size=64)
    manager.pd_node_request_limit_enabled = False
    manager._run_reqs_count_lock = asyncio.Lock()
    manager.run_reqs_count_mark = FakeSharedInt()
    manager.latest_success_infer_time_mark = FakeSharedInt()
    manager.pd_node_shm_req_alloc_timeout_seconds = 20
    manager.shm_req_manager = MagicMock()
    manager.shm_req_manager.async_release_req_index = AsyncMock()
    return manager


def test_shm_req_partial_allocations_are_released_on_failure():
    async def run():
        manager = _manager()
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(side_effect=[3, RuntimeError("failed")])

        with pytest.raises(RuntimeError, match="failed"):
            await manager._alloc_shm_req_indexes(2)

        manager.shm_req_manager.async_release_req_index.assert_awaited_once_with(3)

    asyncio.run(run())


def test_shm_req_allocation_waits_forever_when_local_limit_is_disabled():
    async def run():
        manager = _manager()
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(side_effect=[None, None, 3])

        with patch("lightllm.server.httpserver.manager.asyncio.sleep", new=AsyncMock()) as sleep:
            assert await manager._alloc_shm_req_indexes(1) == [3]

        assert sleep.await_count == 2

    asyncio.run(run())


def test_high_priority_shm_req_allocation_uses_shorter_backoff_even_with_local_limit():
    async def run():
        manager = _manager()
        manager.pd_node_request_limit_enabled = True
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(side_effect=[None, 3])

        with patch("lightllm.server.httpserver.manager.asyncio.sleep", new=AsyncMock()) as sleep:
            assert await manager._alloc_shm_req_indexes(1, pd_high_priority_request=True) == [3]

        assert sleep.await_args_list[0].args[0] == pytest.approx(0.1 * 0.2)

    asyncio.run(run())


def test_pd_high_priority_request_is_inserted_at_router_queue_head():
    queue = BaseQueue.__new__(BaseQueue)
    queue.dp_index = 0
    normal_req = SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=False))
    high_req_1 = SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=True))
    high_req_2 = SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=True))
    queue.waiting_req_list = [normal_req]

    queue.extend([high_req_1, high_req_2])

    assert queue.waiting_req_list == [high_req_1, high_req_2, normal_req]
    assert high_req_1.sample_params.suggested_dp_index == 0
    assert high_req_2.sample_params.suggested_dp_index == 0
