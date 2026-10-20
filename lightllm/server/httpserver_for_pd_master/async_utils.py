"""Cancellation handling for operations that must finish before their owner exits."""

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


async def await_cancellation_safe(awaitable: Awaitable[T]) -> T:
    """Wait through repeated caller cancellations, then propagate cancellation.

    The underlying task is never cancelled by this waiter. This keeps a thread
    backed RPC ordered before any cleanup that depends on its side effects.
    """
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except BaseException as error:
            if cancelled:
                raise asyncio.CancelledError() from error
            raise
    if cancelled:
        raise asyncio.CancelledError()
    return result
