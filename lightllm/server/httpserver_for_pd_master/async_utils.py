"""Cancellation handling for operations that must finish before their owner exits."""

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


async def await_cancellation_safe(awaitable: Awaitable[T]) -> T:
    """Wait through repeated caller cancellations, then propagate cancellation.

    The underlying task is never cancelled by this waiter. This keeps a thread
    backed RPC ordered before any cleanup that depends on its side effects. If
    the operation itself fails, preserve that error even when cancellation was
    also requested, so callers can handle the unknown RPC state safely.
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
        except BaseException:
            # 操作失败比调用者取消更重要。上层必须看到真实异常，不能把一次
            # 状态未知的 RPC 误判成普通取消后继续服务。
            raise
    if cancelled:
        raise asyncio.CancelledError()
    return result
