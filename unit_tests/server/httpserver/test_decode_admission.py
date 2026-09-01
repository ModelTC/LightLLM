import asyncio

import pytest

from lightllm.server.httpserver.decode_admission import (
    DecodeAdmissionController,
    DecodeAdmissionLeaseHandle,
)
from lightllm.utils.error_utils import ServerBusyError


def test_decode_admission_waits_and_grants_in_fifo_order():
    async def run():
        controller = DecodeAdmissionController(capacity=2, max_queued_slots=2, timeout_seconds=1)
        active = await controller.acquire(2)
        first_task = asyncio.create_task(controller.acquire(1))
        second_task = asyncio.create_task(controller.acquire(1))
        await asyncio.sleep(0)

        assert controller.active_slots == 2
        assert controller.queued_slots == 2
        assert not first_task.done()
        assert not second_task.done()

        active.release()
        first = await first_task
        second = await second_task
        assert controller.active_slots == 2

        first.release()
        second.release()
        assert controller.active_slots == 0

    asyncio.run(run())


def test_decode_admission_keeps_strict_fifo_when_head_gang_does_not_fit():
    async def run():
        controller = DecodeAdmissionController(capacity=3, max_queued_slots=3, timeout_seconds=1)
        first_active = await controller.acquire(1)
        second_active = await controller.acquire(1)
        third_active = await controller.acquire(1)
        gang_task = asyncio.create_task(controller.acquire(2))
        await asyncio.sleep(0)
        small_task = asyncio.create_task(controller.acquire(1))
        await asyncio.sleep(0)

        first_active.release()
        await asyncio.sleep(0)
        assert not gang_task.done()
        assert not small_task.done()
        assert controller.active_slots == 2

        second_active.release()
        gang = await gang_task
        assert not small_task.done()
        third_active.release()
        small = await small_task
        assert controller.active_slots == 3

        gang.release()
        small.release()

    asyncio.run(run())


def test_decode_admission_cancellation_removes_waiter():
    async def run():
        controller = DecodeAdmissionController(capacity=1, max_queued_slots=1, timeout_seconds=1)
        active = await controller.acquire(1)
        waiting_task = asyncio.create_task(controller.acquire(1))
        await asyncio.sleep(0)

        waiting_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting_task

        assert controller.queued_request_count == 0
        assert controller.queued_slots == 0
        active.release()
        assert controller.active_slots == 0

    asyncio.run(run())


def test_decode_admission_timeout_removes_waiter():
    async def run():
        controller = DecodeAdmissionController(capacity=1, max_queued_slots=1, timeout_seconds=0.01)
        active = await controller.acquire(1)

        with pytest.raises(ServerBusyError, match="wait timed out"):
            await controller.acquire(1)

        assert controller.queued_request_count == 0
        active.release()

    asyncio.run(run())


def test_decode_admission_rejects_full_queue_and_oversized_gang():
    async def run():
        controller = DecodeAdmissionController(capacity=2, max_queued_slots=1, timeout_seconds=1)
        active = await controller.acquire(2)
        waiting_task = asyncio.create_task(controller.acquire(1))
        await asyncio.sleep(0)

        with pytest.raises(ServerBusyError, match="queue is full"):
            await controller.acquire(1)
        with pytest.raises(ServerBusyError, match="needs 3 Decode slots"):
            await controller.acquire(3)

        waiting_task.cancel()
        await asyncio.gather(waiting_task, return_exceptions=True)
        active.release()

    asyncio.run(run())


def test_decode_admission_cancellation_after_concurrent_grant_releases_lease():
    async def run():
        controller = DecodeAdmissionController(capacity=1, max_queued_slots=1, timeout_seconds=1)
        active = await controller.acquire(1)
        waiting_task = asyncio.create_task(controller.acquire(1))
        await asyncio.sleep(0)

        active.release()
        waiting_task.cancel()
        await asyncio.gather(waiting_task, return_exceptions=True)

        assert controller.active_slots == 0
        assert controller.queued_request_count == 0

    asyncio.run(run())


def test_decode_admission_gang_lease_can_transfer_to_independent_requests():
    async def run():
        controller = DecodeAdmissionController(capacity=3, max_queued_slots=0, timeout_seconds=1)
        gang = await controller.acquire(3)
        first, second, third = gang.split([1, 1, 1])

        first.release()
        assert controller.active_slots == 2
        second.release()
        assert controller.active_slots == 1
        third.release()
        assert controller.active_slots == 0

        with pytest.raises(RuntimeError, match="already been released"):
            gang.split([1, 1, 1])

    asyncio.run(run())


def test_decode_admission_handle_transfers_or_releases_lease_once():
    async def run():
        controller = DecodeAdmissionController(capacity=2, max_queued_slots=0, timeout_seconds=1)
        transferred_handle = DecodeAdmissionLeaseHandle(await controller.acquire(1))
        lease = transferred_handle.take()
        transferred_handle.release()
        assert controller.active_slots == 1
        lease.release()

        abandoned_handle = DecodeAdmissionLeaseHandle(await controller.acquire(1))
        abandoned_handle.release()
        abandoned_handle.release()
        assert controller.active_slots == 0

    asyncio.run(run())
