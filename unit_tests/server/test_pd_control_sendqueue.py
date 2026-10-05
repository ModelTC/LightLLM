import asyncio
import unittest

from lightllm.server.pd_io_struct import PD_Client_Obj


class _GateSocket:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.sent = []
        self.active = 0
        self.max_active = 0

    async def send_bytes(self, payload):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        await self.release.wait()
        self.sent.append(payload)
        self.active -= 1


class _FailSocket:
    async def send_bytes(self, payload):
        raise RuntimeError("wire failure")


def _client(socket):
    return PD_Client_Obj(1, "prefill:8000", "prefill", {}, websocket=socket)


class TestPdControlSendQueue(unittest.TestCase):
    def test_fifo_and_no_overlapping_frames(self):
        async def case():
            socket = _GateSocket()
            client = _client(socket)
            tasks = [asyncio.create_task(client.send_control_message(payload)) for payload in (b"a", b"b", b"c")]
            await asyncio.wait_for(socket.started.wait(), 1)
            socket.release.set()
            await asyncio.gather(*tasks)
            self.assertEqual(socket.sent, [b"a", b"b", b"c"])
            self.assertEqual(socket.max_active, 1)
            self.assertIsNone(client._send_drain_task)

        asyncio.run(case())

    def test_queued_cancellation_releases_frame_before_drain(self):
        async def case():
            socket = _GateSocket()
            client = _client(socket)
            first = asyncio.create_task(client.send_control_message(b"first"))
            await asyncio.wait_for(socket.started.wait(), 1)
            queued = asyncio.create_task(client.send_control_message(b"cancelled"))
            await asyncio.sleep(0)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            socket.release.set()
            await first
            self.assertEqual(socket.sent, [b"first"])
            self.assertIsNone(client._send_drain_task)

        asyncio.run(case())

    def test_active_cancellation_does_not_interrupt_frame_or_next_item(self):
        async def case():
            socket = _GateSocket()
            client = _client(socket)
            active = asyncio.create_task(client.send_control_message(b"active"))
            await asyncio.wait_for(socket.started.wait(), 1)
            later = asyncio.create_task(client.send_control_message(b"later"))
            await asyncio.sleep(0)
            active.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await active
            socket.release.set()
            await later
            self.assertEqual(socket.sent, [b"active", b"later"])
            self.assertEqual(socket.max_active, 1)

        asyncio.run(case())

    def test_failure_wakes_current_and_queued_callers(self):
        async def case():
            client = _client(_FailSocket())
            outcomes = await asyncio.gather(
                client.send_control_message(b"one"), client.send_control_message(b"two"), return_exceptions=True
            )
            self.assertIsInstance(outcomes[0], RuntimeError)
            self.assertIsInstance(outcomes[1], ConnectionError)
            self.assertIsNone(client.websocket)
            self.assertIsNone(client._send_drain_task)
            self.assertFalse(client._send_queue)

        asyncio.run(case())
