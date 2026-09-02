import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.utils.error_utils import ServerBusyError


class FakeSharedInt:
    def __init__(self, value: int = 0):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value: int):
        self.value = value


def _manager(*, running_request_count: int = 0, max_allowed_request_count: int = 1) -> HttpServerManager:
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.args = SimpleNamespace(run_mode="decode", running_max_req_size=64)
    manager.pd_node_request_limit_enabled = True
    manager._run_reqs_count_lock = asyncio.Lock()
    manager.run_reqs_count_mark = FakeSharedInt(running_request_count)
    manager.latest_success_infer_time_mark = FakeSharedInt()
    manager.qps_recorder = MagicMock()
    manager.qps_recorder.get_max_allowed_request_count.return_value = max_allowed_request_count
    manager.shm_req_manager = MagicMock()
    manager.shm_req_manager.async_release_req_index = AsyncMock()
    return manager


def test_pd_node_rejects_request_when_running_concurrency_exceeds_limit():
    async def run():
        manager = _manager(running_request_count=2, max_allowed_request_count=1)
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(return_value=7)

        with pytest.raises(ServerBusyError):
            await manager._alloc_shm_req_indexes(1)

        manager.shm_req_manager.async_alloc_req_index.assert_not_awaited()

    asyncio.run(run())


def test_pd_node_allows_request_when_running_concurrency_equals_limit():
    async def run():
        manager = _manager(running_request_count=1, max_allowed_request_count=1)
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(return_value=7)

        indexes = await manager._alloc_shm_req_indexes(1)

        assert indexes == [7]
        manager.qps_recorder.get_max_allowed_request_count.assert_called_once_with()

    asyncio.run(run())


def test_pd_high_priority_split_continuation_bypasses_running_concurrency_limit():
    async def run():
        manager = _manager(running_request_count=2, max_allowed_request_count=1)
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(side_effect=[None, 7])

        with patch("lightllm.server.httpserver.manager.asyncio.sleep", new=AsyncMock()):
            indexes = await manager._alloc_shm_req_indexes(
                1,
                pd_high_priority_request=True,
            )

        assert indexes == [7]
        manager.qps_recorder.get_max_allowed_request_count.assert_not_called()

    asyncio.run(run())


def test_admitted_request_waits_for_shm_req_without_timeout():
    async def run():
        manager = _manager()
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(side_effect=[None, None, 7])

        with patch("lightllm.server.httpserver.manager.asyncio.sleep", new=AsyncMock()) as sleep:
            indexes = await manager._alloc_shm_req_indexes(1)

        assert indexes == [7]
        assert sleep.await_count == 2
        manager.shm_req_manager.async_release_req_index.assert_not_awaited()

    asyncio.run(run())


def test_shm_req_partial_allocations_are_released_on_failure():
    async def run():
        manager = _manager()
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(side_effect=[3, RuntimeError("failed")])

        with pytest.raises(RuntimeError, match="failed"):
            await manager._alloc_shm_req_indexes(2)

        manager.shm_req_manager.async_release_req_index.assert_awaited_once_with(3)

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
