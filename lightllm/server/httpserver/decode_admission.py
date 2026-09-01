from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

from lightllm.utils.error_utils import ServerBusyError


@dataclass(slots=True)
class _Waiter:
    slots: int
    enqueue_time: float
    future: asyncio.Future


class DecodeAdmissionLease:
    """A fixed number of request slots reserved on one Decode node."""

    def __init__(self, controller: "DecodeAdmissionController", slots: int, waited_seconds: float) -> None:
        self._controller = controller
        self.slots = slots
        self.waited_seconds = waited_seconds
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller._release(self.slots)

    def split(self, slots: list[int]) -> list["DecodeAdmissionLease"]:
        """Transfer this reservation into independently releasable child leases."""
        if self._released:
            raise RuntimeError("Decode admission lease has already been released")
        if not slots or any(slot_count < 1 for slot_count in slots) or sum(slots) != self.slots:
            raise ValueError("child lease slots must be positive and sum to the parent lease")

        self._released = True
        return [DecodeAdmissionLease(self._controller, slot_count, self.waited_seconds) for slot_count in slots]


class DecodeAdmissionLeaseHandle:
    """Transfers a pre-acquired lease into an asynchronously started generator."""

    def __init__(self, lease: DecodeAdmissionLease) -> None:
        self._lease: Optional[DecodeAdmissionLease] = lease

    def take(self) -> DecodeAdmissionLease:
        if self._lease is None:
            raise RuntimeError("Decode admission lease has already been taken")
        lease = self._lease
        self._lease = None
        return lease

    def release(self) -> None:
        if self._lease is None:
            return
        self._lease.release()
        self._lease = None


class DecodeAdmissionController:
    """A bounded, cancellable FIFO for request slots owned by one Decode node."""

    def __init__(
        self,
        capacity: int,
        max_queued_slots: int,
        timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError("decode admission capacity must be positive")
        if max_queued_slots < 0:
            raise ValueError("decode admission queue size must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("decode admission timeout must be positive")

        self.capacity = capacity
        self.max_queued_slots = max_queued_slots
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self._active_slots = 0
        self._queued_slots = 0
        self._waiters: Deque[_Waiter] = deque()

    @property
    def active_slots(self) -> int:
        return self._active_slots

    @property
    def queued_slots(self) -> int:
        return self._queued_slots

    @property
    def queued_request_count(self) -> int:
        return len(self._waiters)

    async def acquire(self, slots: int) -> DecodeAdmissionLease:
        if slots < 1:
            raise ValueError("decode admission slots must be positive")
        if slots > self.capacity:
            raise ServerBusyError(f"request needs {slots} Decode slots, but the node capacity is {self.capacity}")

        if not self._waiters and self._active_slots + slots <= self.capacity:
            return self._activate(slots, waited_seconds=0.0)

        if self._queued_slots + slots > self.max_queued_slots:
            raise ServerBusyError("Decode node admission queue is full")

        waiter = _Waiter(
            slots=slots,
            enqueue_time=self._clock(),
            future=asyncio.get_running_loop().create_future(),
        )
        self._waiters.append(waiter)
        self._queued_slots += slots
        self._drain()

        try:
            return await asyncio.wait_for(asyncio.shield(waiter.future), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            lease = self._remove_waiter_or_take_lease(waiter)
            if lease is not None:
                lease.release()
            raise ServerBusyError("Decode node admission queue wait timed out") from exc
        except BaseException:
            lease = self._remove_waiter_or_take_lease(waiter)
            if lease is not None:
                lease.release()
            raise

    def _activate(self, slots: int, waited_seconds: float) -> DecodeAdmissionLease:
        self._active_slots += slots
        return DecodeAdmissionLease(self, slots, waited_seconds)

    def _release(self, slots: int) -> None:
        self._active_slots -= slots
        if self._active_slots < 0:
            raise RuntimeError("Decode admission active slot count became negative")
        self._drain()

    def _drain(self) -> None:
        while self._waiters:
            waiter = self._waiters[0]
            if waiter.future.done():
                self._remove_waiter(waiter)
                continue
            if self._active_slots + waiter.slots > self.capacity:
                break

            self._remove_waiter(waiter)
            waiter.future.set_result(
                self._activate(
                    waiter.slots,
                    waited_seconds=max(0.0, self._clock() - waiter.enqueue_time),
                )
            )

    def _remove_waiter(self, waiter: _Waiter) -> bool:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            return False
        self._queued_slots -= waiter.slots
        return True

    def _remove_waiter_or_take_lease(self, waiter: _Waiter) -> Optional[DecodeAdmissionLease]:
        if self._remove_waiter(waiter):
            waiter.future.cancel()
            self._drain()
            return None
        if waiter.future.done() and not waiter.future.cancelled():
            result = waiter.future.result()
            if isinstance(result, DecodeAdmissionLease):
                return result
        return None
