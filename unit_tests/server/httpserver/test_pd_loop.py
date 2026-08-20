import asyncio
import pickle
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightllm.server.httpserver import pd_loop
from lightllm.server.pd_io_struct import ObjType


class _Socket:
    def setsockopt(self, *args):
        pass


class _WebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.transport = SimpleNamespace(get_extra_info=lambda _: _Socket())

    async def send(self, data):
        pass

    async def recv(self):
        await asyncio.sleep(0)
        try:
            return next(self.messages)
        except StopIteration:
            await asyncio.Future()


class _Connection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, traceback):
        pass


def test_abort_cancels_only_the_matching_pending_generation_task():
    async def run():
        request_ids = (4440000, 4440001)
        started = {request_id: asyncio.Event() for request_id in request_ids}
        cancelled = {request_id: asyncio.Event() for request_id in request_ids}

        async def process_generate(**kwargs):
            request_id = kwargs["sampling_params"].group_request_id
            started[request_id].set()
            try:
                await asyncio.Future()
            finally:
                cancelled[request_id].set()

        messages = [
            pickle.dumps(
                (
                    ObjType.REQ,
                    ("prompt", SimpleNamespace(group_request_id=request_id), None),
                )
            )
            for request_id in request_ids
        ]
        messages.append(pickle.dumps((ObjType.ABORT, request_ids[0])))
        websocket = _WebSocket(messages)

        manager = SimpleNamespace(
            args=SimpleNamespace(pd_node_id=1),
            host_ip="127.0.0.2",
            pd_mode=SimpleNamespace(value="prefill"),
            abort=AsyncMock(return_value=False),
        )
        pd_master_obj = SimpleNamespace(host_ip_port="127.0.0.3:1234")

        with (
            patch.object(pd_loop.websockets, "connect", return_value=_Connection(websocket)),
            patch.object(pd_loop, "get_shm_port_args", return_value=SimpleNamespace(port=1234)),
            patch.object(pd_loop, "_pd_process_generate", side_effect=process_generate),
        ):
            handler_task = asyncio.create_task(pd_loop._pd_handle_task(manager, pd_master_obj))
            try:
                await asyncio.wait_for(started[request_ids[1]].wait(), timeout=1)
                await asyncio.wait_for(cancelled[request_ids[0]].wait(), timeout=1)
                await asyncio.sleep(0)

                manager.abort.assert_awaited_once_with(request_ids[0])
                assert not cancelled[request_ids[1]].is_set()
            finally:
                handler_task.cancel()
                with suppress(asyncio.CancelledError):
                    await handler_task

            assert cancelled[request_ids[1]].is_set()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_pd_process_generate_propagates_cancellation():
    async def run():
        started = asyncio.Event()

        class _Manager:
            args = SimpleNamespace(run_mode="prefill")

            async def generate(self, **kwargs):
                started.set()
                await asyncio.Future()
                yield

        generation_task = asyncio.create_task(
            pd_loop._pd_process_generate(
                manager=_Manager(),
                prompt="prompt",
                sampling_params=SimpleNamespace(group_request_id=4440000),
                multimodal_params=None,
                forwarding_queue=AsyncMock(),
                pd_upload_websocket=AsyncMock(),
                pd_event=asyncio.Event(),
            )
        )
        await started.wait()
        generation_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await generation_task

    asyncio.run(asyncio.wait_for(run(), timeout=2))
