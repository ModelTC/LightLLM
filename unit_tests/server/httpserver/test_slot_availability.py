import asyncio
from types import SimpleNamespace
import pytest
import torch
from lightllm.server.httpserver.slot_availability import RequestSlotAvailability
from lightllm.server.core.objs.shm_req_manager import ShmReqManager

from lightllm.server.httpserver.manager import HttpServerManager
import lightllm.server.httpserver.manager as module


class Pool(ShmReqManager):
    def __init__(self, indices=()):
        self.free = list(indices)
        self.claimed = set()
        self.empty = asyncio.Event()
        self.fail_release = False

    def alloc_req_index(self):
        if not self.free:
            self.empty.set()
            return None
        index = self.free.pop(0)
        assert index not in self.claimed
        self.claimed.add(index)
        return index

    def alloc_req_indexes(self, count):
        if len(self.free) < count:
            self.empty.set()
            return None
        return [self.alloc_req_index() for _ in range(count)]

    def release_req_index(self, index):
        if self.fail_release:
            raise RuntimeError("test release failed")
        self.claimed.discard(index)
        self.free.append(index)

    async def async_put_back_req_obj(self, req):
        pass

    def __del__(self):
        pass


def manager(pool, availability=True):
    result = HttpServerManager.__new__(HttpServerManager)
    result.shm_req_manager = pool
    result._request_slot_availability = RequestSlotAvailability(availability)
    result.is_multinode_tp_slave = False
    result.args = SimpleNamespace(running_max_req_size=8, run_mode="normal")
    return result


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def test_original_async_allocator_keeps_immediate_none_contract():
    async def scenario():
        assert await Pool().async_alloc_req_index() is None

    asyncio.run(scenario())


def test_generation_change_before_wait_is_observed():
    async def scenario():
        notifier = RequestSlotAvailability()
        generation = notifier.generation
        notifier.notify_release()
        assert await notifier.wait(generation, 10)

    asyncio.run(scenario())


def test_waiting_task_is_notified_after_release():
    async def scenario():
        notifier = RequestSlotAvailability()
        task = asyncio.create_task(notifier.wait(0, 10))
        await asyncio.sleep(0)
        notifier.notify_release()
        assert await task

    asyncio.run(scenario())


def test_unobserved_external_release_uses_timeout_fallback():
    async def scenario():
        notifier = RequestSlotAvailability()
        assert not await notifier.wait(notifier.generation, 0.001)

    asyncio.run(scenario())


def test_actual_http_retry_observes_release_between_check_and_wait():
    async def scenario():
        pool = Pool()
        http = manager(pool)
        availability = http._request_slot_availability
        original = availability.wait

        async def release_before_wait(generation, timeout):
            await pool.async_release_req_index(7)
            http._notify_req_slot_release()
            return await original(generation, timeout)

        availability.wait = release_before_wait
        assert await http._alloc_shm_req_indexes(1) == [7]
        assert pool.claimed == {7}

    asyncio.run(scenario())


def test_recycler_notifies_only_after_actual_release():
    async def scenario():
        pool = Pool()
        pool.claimed = {7}
        http = manager(pool)
        http.recycle_event = asyncio.Event()
        req = SimpleNamespace(request_id=8, index_in_shm_mem=7)
        group = SimpleNamespace(group_req_id=8, shm_req_objs=[req], multimodal_params=None)
        http.req_id_to_out_inf = {8: SimpleNamespace(can_release=lambda: True, group_req_objs=group)}

        async def release_media(params):
            pass

        http._release_multimodal_resources = release_media
        waiter = asyncio.create_task(http._alloc_shm_req_indexes(1))
        await pool.empty.wait()
        recycler = asyncio.create_task(http.recycle_resource_loop())
        http.recycle_event.set()
        try:
            assert await waiter == [7]
            assert not http.req_id_to_out_inf
            assert http._request_slot_availability.generation == 1
        finally:
            recycler.cancel()
            with pytest.raises(asyncio.CancelledError):
                await recycler

    asyncio.run(scenario())


def test_http_cancellation_keeps_unallocated_group_free():
    async def scenario():
        pool = Pool([3])
        http = manager(pool)
        task = asyncio.create_task(http._alloc_shm_req_indexes(2))
        await pool.empty.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.free == [3] and not pool.claimed
        assert http._request_slot_availability.generation == 0

    asyncio.run(scenario())


def test_waiter_cancellation_does_not_consume_notification():
    async def scenario():
        notifier = RequestSlotAvailability()
        task = asyncio.create_task(notifier.wait(0, 10))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        notifier.notify_release()
        assert await notifier.wait(0, 10)

    asyncio.run(scenario())


def test_multiple_http_waiters_do_not_duplicate_allocation():
    async def scenario():
        pool = Pool()
        http = manager(pool)
        tasks = [asyncio.create_task(http._alloc_shm_req_indexes(1)) for _ in range(2)]
        await pool.empty.wait()
        await pool.async_release_req_index(4)
        http._notify_req_slot_release()
        await pool.async_release_req_index(5)
        http._notify_req_slot_release()
        assert {result[0] for result in await asyncio.gather(*tasks)} == {4, 5}
        assert pool.claimed == {4, 5}

    asyncio.run(scenario())


def test_failed_release_does_not_increment_generation():
    async def scenario():
        pool = Pool([3])
        http = manager(pool)
        pool.fail_release = True

        task = asyncio.create_task(http._alloc_shm_req_indexes(2))
        await pool.empty.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert http._request_slot_availability.generation == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        dict(run_mode="prefill"),
        dict(run_mode="decode"),
        dict(dp=2),
        dict(nnodes=2),
        dict(enable_rl=True),
        dict(httpserver_workers=2),
    ],
)
def test_unverified_modes_keep_legacy_backoff(change):
    args = SimpleNamespace(run_mode="normal", dp=1, nnodes=1, enable_rl=False, httpserver_workers=1)
    for key, value in change.items():
        setattr(args, key, value)
    notifier = RequestSlotAvailability.from_args(args)
    assert not notifier.enabled
    notifier.notify_release()
    assert notifier.generation == 0


@pytest.mark.parametrize("priority,expected", [(False, 0.1), (True, 0.02)])
def test_original_priority_backoff_interval_is_preserved(priority, expected):
    async def scenario():
        pool = Pool()
        http = manager(pool, False)
        observed = []

        async def wait_and_release(generation, timeout):
            observed.append(timeout)
            await pool.async_release_req_index(7)
            return False

        http._request_slot_availability.wait = wait_and_release
        assert await http._alloc_shm_req_indexes(1, high_priority_request=priority) == [7]
        assert observed == pytest.approx([expected])

    asyncio.run(scenario())


def test_existing_timeout_preserves_unallocated_group():
    async def scenario():
        pool = Pool([3])
        http = manager(pool, False)
        http.pd_node_request_limit_enabled = True
        http.pd_node_shm_req_alloc_timeout_seconds = 0
        http.args = SimpleNamespace(run_mode="prefill", running_max_req_size=8)
        with pytest.raises(module.ServerBusyError):
            await http._alloc_shm_req_indexes(2, pd_node_resource_wait_timeout_seconds=0)
        assert pool.free == [3] and not pool.claimed

    asyncio.run(scenario())


@pytest.mark.parametrize("mode,workers,enabled", [("normal", 1, True), ("decode", 1, False), ("normal", 2, False)])
def test_real_constructor_initializes_notifier_before_external_io(monkeypatch, mode, workers, enabled):
    args = SimpleNamespace(run_mode=mode, dp=1, nnodes=1, enable_rl=False, httpserver_workers=workers)

    def stop_before_external_io():
        raise RuntimeError("stop_before_external_io")

    monkeypatch.setattr(module, "get_shm_port_args", stop_before_external_io)
    http = HttpServerManager.__new__(HttpServerManager)
    with pytest.raises(RuntimeError, match="stop_before_external_io"):
        http.__init__(args)
    assert http._request_slot_availability.enabled is enabled
