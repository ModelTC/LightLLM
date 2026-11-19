"""Decode trans packs the KV handshake and pd_report forwards it to the PD master."""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo, PDUpKVStatus
from lightllm.server.pd_report.manager import PdReportManager
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process


def test_decode_trans_sends_the_handshake_to_pd_report():
    sent = []
    module = decode_trans_process._DecodeTransModule.__new__(decode_trans_process._DecodeTransModule)
    module.args = SimpleNamespace(pd_node_id=7)
    module.pd_report_push = SimpleNamespace(send=lambda message: sent.append(pickle.dumps(message)))
    module.transporter = SimpleNamespace(
        agent_name="7_1",
        agent_metadata=b"meta",
        num_pages=3,
        local_page_mem_desc=b"desc",
    )
    task = SimpleNamespace(request_id=8, pd_master_node_id=100, start_kv_index=6)

    module._push_up_kv_status(task=task)

    status = pickle.loads(sent[0])
    assert status == PDUpKVStatus(
        group_request_id=8,
        pd_master_node_id=100,
        pd_kv_trans_params=pickle.dumps(
            PDDecodeNodeInfo(
                decode_node_id=7,
                pd_master_node_id=100,
                agent_name="7_1",
                agent_metadata=b"meta",
                num_pages=3,
                page_reg_desc=b"desc",
                request_id=8,
                ready_kv_len=6,
            )
        ),
    )


def _deliver(manager: PdReportManager, messages: list) -> None:
    pending = list(messages)

    async def recv():
        if pending:
            return pending.pop(0)
        raise asyncio.CancelledError

    manager._pull = SimpleNamespace(recv=recv)

    async def run():
        try:
            await manager._recv_reports()
        except asyncio.CancelledError:
            return

    asyncio.run(run())


def test_pd_report_connects_with_protocol_ping(monkeypatch):
    import websockets

    seen = {}

    class IdleSocket:
        def __init__(self):
            self.transport = SimpleNamespace(get_extra_info=lambda key: None)

        async def send(self, payload):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    def connect(**kwargs):
        seen.update(kwargs)
        return IdleSocket()

    monkeypatch.setattr(websockets, "connect", connect)

    async def run():
        manager = PdReportManager(args=SimpleNamespace(), pull=None)
        master = SimpleNamespace(node_id=1, host_ip_port="127.0.0.1:1")
        task = asyncio.create_task(manager._serve_master(master=master, outgoing=asyncio.Queue()))
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert seen["ping_interval"] == 60
    assert seen["ping_timeout"] == 60


def test_pd_report_routes_the_handshake_by_master_id():
    manager = PdReportManager(args=SimpleNamespace(), pull=None)
    manager._queues[100] = asyncio.Queue()
    manager._queues[200] = asyncio.Queue()
    status = PDUpKVStatus(9, 100, b"metadata")

    _deliver(manager, [status])

    assert manager._queues[100].get_nowait() == (ObjType.PD_UPLOAD_KV_STATUS, status)
    assert manager._queues[200].empty()


def test_pd_report_holds_a_handshake_until_the_master_connects():
    manager = PdReportManager(args=SimpleNamespace(), pull=None)
    manager._queues[100] = asyncio.Queue()
    first = PDUpKVStatus(8, 200, b"metadata")
    second = PDUpKVStatus(8, 300, b"metadata")
    _deliver(manager, [first, second])
    assert manager._queues[100].empty()
    assert manager._queues[200].get_nowait() == (ObjType.PD_UPLOAD_KV_STATUS, first)
    assert manager._queues[300].get_nowait() == (ObjType.PD_UPLOAD_KV_STATUS, second)


def test_failed_report_send_is_retried_after_reconnect(monkeypatch):
    sent = []
    attempts = {"count": 0}

    class FlakySocket:
        def __init__(self):
            self.transport = SimpleNamespace(get_extra_info=lambda key: None)

        async def send(self, payload):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionError("send failed")
            sent.append(payload)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("websockets.connect", lambda **kwargs: FlakySocket())

    original_sleep = asyncio.sleep

    async def no_sleep(delay):
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    async def run():
        manager = PdReportManager(args=SimpleNamespace(), pull=None)
        outgoing = asyncio.Queue()
        payload = (ObjType.PD_UPLOAD_KV_STATUS, PDUpKVStatus(8, 100, b"metadata"))
        outgoing.put_nowait(payload)
        task = asyncio.create_task(
            manager._serve_master(master=SimpleNamespace(node_id=100, host_ip_port="127.0.0.1:1"), outgoing=outgoing)
        )
        for _ in range(50):
            if sent:
                break
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert pickle.loads(sent[0]) == (ObjType.PD_UPLOAD_KV_STATUS, PDUpKVStatus(8, 100, b"metadata"))


def test_master_reports_kv_status_from_the_register_channel(monkeypatch):
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(config_server_host=None, config_server_port=None)
        manager.update_req_status = AsyncMock()

        async def idle(self):
            await asyncio.Event().wait()

        monkeypatch.setattr(HttpServerManagerForPDMaster, "timer_log", idle)
        task = asyncio.create_task(manager.handle_loop())
        try:
            for _ in range(50):
                if getattr(manager, "infos_queues", None) is not None:
                    break
                await asyncio.sleep(0.01)
            status = PDUpKVStatus(8, 100, b"metadata")
            await manager.infos_queues.put((ObjType.PD_UPLOAD_KV_STATUS, status))
            for _ in range(50):
                if manager.update_req_status.await_count:
                    break
                await asyncio.sleep(0.01)
            manager.update_req_status.assert_awaited_once_with(status)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
