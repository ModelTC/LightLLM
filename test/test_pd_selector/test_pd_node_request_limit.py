import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightllm.server.core.objs import SamplingParams
from lightllm.server.httpserver.manager import HttpServerManager, ReqStatus
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.router.req_queue.dp_base_queue import DpQueue
from lightllm.utils.error_utils import ServerBusyError


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


def test_pd_high_priority_timeout_is_internal_and_defaults_to_zero():
    sampling_params = SamplingParams()
    sampling_params.init(
        None,
        pd_high_priority_request=True,
        pd_high_priority_request_time_out_seconds=99,
    )

    assert sampling_params.pd_high_priority_request is False
    assert sampling_params.pd_high_priority_request_time_out_seconds == 0


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


def test_high_priority_shm_req_allocation_uses_master_timeout_with_local_limit():
    async def run():
        manager = _manager()
        manager.pd_node_request_limit_enabled = True
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(return_value=None)

        with (
            patch("lightllm.server.httpserver.manager.time.monotonic", side_effect=[100, 161]),
            pytest.raises(ServerBusyError, match="within 60 seconds"),
        ):
            await manager._alloc_shm_req_indexes(
                1,
                pd_high_priority_request=True,
                pd_high_priority_request_time_out_seconds=60,
            )

    asyncio.run(run())


def test_high_priority_shm_req_allocation_does_not_shorten_local_timeout():
    async def run():
        manager = _manager()
        manager.pd_node_request_limit_enabled = True
        manager.pd_node_shm_req_alloc_timeout_seconds = 80
        manager.shm_req_manager.async_alloc_req_index = AsyncMock(return_value=None)

        with (
            patch("lightllm.server.httpserver.manager.time.monotonic", side_effect=[100, 181]),
            pytest.raises(ServerBusyError, match="within 80 seconds"),
        ):
            await manager._alloc_shm_req_indexes(
                1,
                pd_high_priority_request=True,
                pd_high_priority_request_time_out_seconds=60,
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("infer_start_time", "local_timeout_seconds", "high_priority_timeout_seconds", "expected"),
    [(0, 20, 0, True), (0, 20, 60, True), (0, 80, 60, False), (1, 20, 60, False)],
)
def test_high_priority_router_wait_uses_master_timeout(
    infer_start_time,
    local_timeout_seconds,
    high_priority_timeout_seconds,
    expected,
):
    req_status = ReqStatus.__new__(ReqStatus)
    req_status.group_req_objs = SimpleNamespace(
        shm_req_objs=[
            SimpleNamespace(
                infer_start_time=infer_start_time,
                router_arrival_time=100,
                sample_params=SimpleNamespace(
                    pd_high_priority_request=True,
                    pd_high_priority_request_time_out_seconds=high_priority_timeout_seconds,
                ),
            )
        ]
    )

    with patch("lightllm.server.httpserver.manager.time.monotonic", return_value=161):
        assert req_status.has_timed_out_waiting_for_inference(local_timeout_seconds) is expected


def test_pd_high_priority_request_is_inserted_before_first_normal_request():
    queue = BaseQueue.__new__(BaseQueue)
    queue.dp_index = 0
    earlier_high_req = SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=True))
    normal_req = SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=False))
    high_req_1 = SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=True))
    high_req_2 = SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=True))
    queue.waiting_req_list = [earlier_high_req, normal_req]

    queue.extend([high_req_1, high_req_2])

    assert queue.waiting_req_list == [earlier_high_req, high_req_1, high_req_2, normal_req]
    assert high_req_1.sample_params.suggested_dp_index == 0
    assert high_req_2.sample_params.suggested_dp_index == 0


def test_pd_high_priority_request_group_keeps_fifo_order_while_waiting_for_dp_index():
    queue = DpQueue.__new__(DpQueue)
    queue.dp_size_in_node = 2
    earlier_high_group = [SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=True))]
    normal_group = [SimpleNamespace(sample_params=SimpleNamespace(pd_high_priority_request=False))]
    new_high_group = [
        SimpleNamespace(
            sample_params=SimpleNamespace(
                pd_high_priority_request=True,
                suggested_dp_index=-1,
            )
        )
    ]
    queue.reqs_waiting_for_dp_index = [earlier_high_group, normal_group]

    queue.extend(new_high_group)

    assert queue.reqs_waiting_for_dp_index == [earlier_high_group, new_high_group, normal_group]
