"""KV status routing through the master owners discovered by P/D nodes."""

import asyncio
import pickle
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from lightllm.server.pd_io_struct import PD_Master_Obj, PDUpKVStatus
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import up_status


def _manager():
    manager = up_status.UpStatusManager.__new__(up_status.UpStatusManager)
    manager.args = SimpleNamespace(pd_node_id=20)
    manager.id_to_handle_queue = {}
    return manager


def _websocket(send):
    return SimpleNamespace(
        send=send,
        recv=AsyncMock(),
        transport=SimpleNamespace(get_extra_info=lambda _key: MagicMock()),
    )


def test_status_is_sent_directly_without_control_frames_or_reply(monkeypatch):
    async def run():
        manager = _manager()
        status = PDUpKVStatus(8, 100, b"decode metadata")
        manager.id_to_handle_queue[100] = asyncio.Queue()
        manager.id_to_handle_queue[100].put_nowait(status)
        sent = []
        status_sent = asyncio.Event()

        async def send(data):
            sent.append(pickle.loads(data))
            status_sent.set()

        websocket = _websocket(send)

        @asynccontextmanager
        async def connect(uri):
            assert uri == "ws://master:8100/kv_move_status"
            yield websocket

        monkeypatch.setattr(up_status.websockets, "connect", connect)
        task = asyncio.create_task(manager.up_kv_status_task(PD_Master_Obj(100, "master:8100")))
        try:
            await asyncio.wait_for(status_sent.wait(), timeout=1)
            assert sent == [status]
            websocket.recv.assert_not_called()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_topology_refresh_connects_all_new_workers_and_retires_previous_ids(monkeypatch):
    from lightllm.server.httpserver import pd_loop

    async def run():
        manager = _manager()
        responses = asyncio.Queue()
        started = asyncio.Queue()
        retired = set()
        children = []
        original_sleep = asyncio.sleep

        async def discover(_args):
            return await responses.get()

        async def dispatch():
            children.append(asyncio.current_task())
            await asyncio.Event().wait()

        async def connect(master):
            children.append(asyncio.current_task())
            started.put_nowait(master.node_id)
            try:
                await asyncio.Event().wait()
            finally:
                retired.add(master.node_id)

        async def skip_poll_interval(seconds):
            assert seconds == 30, "Topology refresh unexpectedly failed"
            await original_sleep(0)

        monkeypatch.setattr(pd_loop, "_get_pd_master_objs", discover)
        monkeypatch.setattr(manager, "dispatch_task_loop", dispatch)
        monkeypatch.setattr(manager, "up_kv_status_task", connect)
        monkeypatch.setattr(up_status.asyncio, "sleep", skip_poll_interval)
        task = asyncio.create_task(manager.task_loop())
        try:
            responses.put_nowait({node_id: PD_Master_Obj(node_id, f"master:{node_id}") for node_id in (100, 101)})
            assert {await asyncio.wait_for(started.get(), timeout=1) for _ in range(2)} == {100, 101}
            responses.put_nowait({node_id: PD_Master_Obj(node_id, f"master:{node_id}") for node_id in (200, 201)})
            assert {await asyncio.wait_for(started.get(), timeout=1) for _ in range(2)} == {200, 201}
            assert retired == {100, 101}
            assert set(manager.id_to_handle_queue) == {200, 201}
        finally:
            task.cancel()
            for child in children:
                child.cancel()
            await asyncio.gather(task, *children, return_exceptions=True)

    asyncio.run(run())
