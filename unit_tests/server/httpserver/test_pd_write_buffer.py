"""Exercise PD write limits and direct-send cancellation over real TCP."""

import asyncio
from contextlib import asynccontextmanager
import os
import socket

from fastapi import FastAPI, WebSocket
from hypercorn.asyncio import run
from hypercorn.config import Config
from hypercorn.utils import wrap_app
import pytest
import websockets

from lightllm.utils.websocket_utils import PD_WEBSOCKET_WRITE_LIMIT, configure_pd_master_write_buffer


@pytest.mark.parametrize("compression", [None, "deflate"])
def test_pd_write_limits_and_cancelled_frame_preserve_following_messages(monkeypatch, compression):
    monkeypatch.setattr(run, "TCPServer", run.TCPServer)

    async def exercise():
        connections = []
        shutdown, ready, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        result = asyncio.get_running_loop().create_future()
        payload = os.urandom(256 * 1024)

        @asynccontextmanager
        async def lifespan(_app):
            configure_pd_master_write_buffer()
            configured_server = run.TCPServer

            class RecordingTCPServer(configured_server):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    connections.append(self)

            run.TCPServer = RecordingTCPServer
            ready.set()
            yield

        app = FastAPI(lifespan=lifespan)

        @app.websocket("/pd_register")
        async def exchange(websocket: WebSocket):
            sending = None
            try:
                await websocket.accept()
                assert await websocket.receive_bytes() == b"start"
                writer = connections[-1].writer
                assert writer.transport.get_write_buffer_limits() == (8 * 1024, 32 * 1024)
                writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
                # The default 32 KiB watermark is smaller than this payload. Raise it
                # only inside the test so the first send can sit in the asyncio buffer.
                writer.transport.set_write_buffer_limits(high=1024 * 1024, low=256 * 1024)

                await websocket.send_bytes(payload)
                assert writer.transport.get_write_buffer_size() > 0

                writer.transport.set_write_buffer_limits(high=4096, low=1024)
                sending = asyncio.create_task(websocket.send_bytes(payload))

                async def wait_for_backpressure():
                    while writer.transport.get_write_buffer_size() <= 4096:
                        assert not sending.done()
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait_for_backpressure(), 3)
                assert not sending.done()
                sending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await sending
                cancelled.set()
                await websocket.send_bytes(b"abort")
                assert await websocket.receive_bytes() == b"ack"
                await websocket.close()
                result.set_result(None)
            except BaseException as error:
                if not result.done():
                    result.set_exception(error)
                raise
            finally:
                if sending is not None and not sending.done():
                    sending.cancel()
                    await asyncio.gather(sending, return_exceptions=True)

        config = Config()
        config.bind = ["127.0.0.1:0"]
        config.accesslog = config.errorlog = None
        sockets = config.create_sockets()
        port = sockets.insecure_sockets[0].getsockname()[1]
        server = asyncio.create_task(
            run.worker_serve(
                wrap_app(app, config.wsgi_max_body_size, "asgi"),
                config,
                sockets=sockets,
                shutdown_trigger=shutdown.wait,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), 3)
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/pd_register",
                compression=compression,
                write_limit=PD_WEBSOCKET_WRITE_LIMIT,
                max_size=2 * len(payload),
                ping_interval=None,
                proxy=None,
            ) as peer:
                assert peer.transport.get_write_buffer_limits() == (8 * 1024, 32 * 1024)
                peer.transport.pause_reading()
                try:
                    await peer.send(b"start")
                    await asyncio.wait_for(cancelled.wait(), 5)
                finally:
                    peer.transport.resume_reading()
                assert await asyncio.wait_for(peer.recv(), 5) == payload
                assert await asyncio.wait_for(peer.recv(), 5) == payload
                assert await asyncio.wait_for(peer.recv(), 5) == b"abort"
                await peer.send(b"ack")
                await asyncio.wait_for(result, 5)
        finally:
            shutdown.set()
            await asyncio.wait_for(server, 5)
            if result.done() and not result.cancelled():
                result.exception()

    asyncio.run(exercise())
