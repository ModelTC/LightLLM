import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.httpserver.task_group import HttpServerTasks


def test_shutdown_waits_for_owned_tasks_and_rejects_new_work():
    async def run():
        tasks = HttpServerTasks()
        started = asyncio.Event()
        stopped = []

        async def worker():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)

        task = tasks.create(worker())
        await started.wait()
        await tasks.close()
        assert task.cancelled() and stopped == [True]
        assert not tasks.tasks
        await tasks.close()
        with pytest.raises(RuntimeError, match="closing"):
            tasks.create(worker())

    asyncio.run(run())


def test_recycler_stops_if_wait_completion_consumes_cancellation(monkeypatch):
    from lightllm.server.httpserver.manager import HttpServerManager

    async def run():
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.recycle_event = asyncio.Event()
        manager.req_id_to_out_inf = {}
        entered = asyncio.Event()

        async def completed_during_cancel(coroutine, timeout):
            coroutine.close()
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return True

        monkeypatch.setattr(asyncio, "wait_for", completed_during_cancel)
        task = manager.background_tasks.create(manager.recycle_resource_loop())
        await entered.wait()
        await manager.background_tasks.close()
        assert task.done() and not task.cancelled()
        assert task.exception() is None

    asyncio.run(run())


def test_http_shutdown_closes_background_tasks_before_children(monkeypatch):
    from lightllm.server import api_http
    import psutil

    async def run():
        close = AsyncMock()
        monkeypatch.setattr(api_http.g_objs, "httpserver_manager", SimpleNamespace(close=close))
        monkeypatch.setattr(api_http.asyncio, "sleep", AsyncMock())

        def children(recursive):
            close.assert_awaited_once()
            return []

        monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(children=children))
        await api_http.shutdown()

    asyncio.run(run())
