"""Decode trans packs the KV handshake and httpserver forwards it to the PD master."""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock

from lightllm.server.httpserver.pd_loop import PDMasterForwarder
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo, PDUpKVStatus
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_trans_process


def test_decode_trans_packs_the_handshake_for_httpserver():
    sent = []
    module = decode_trans_process._DecodeTransModule.__new__(decode_trans_process._DecodeTransModule)
    module.args = SimpleNamespace(pd_node_id=7)
    module.pd_httpserver_push = SimpleNamespace(send=lambda message: sent.append(pickle.dumps(message)))
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


def _run_one_ipc_message(monkeypatch, forwarder, status):
    delivered = False

    class Pull:
        def bind(self) -> None:
            return None

        async def recv(self):
            nonlocal delivered
            if not delivered:
                delivered = True
                return status
            await asyncio.Event().wait()

        def close(self) -> None:
            return None

    monkeypatch.setattr("lightllm.server.pd_httpserver_ipc.PDHttpserverIpc.Pull", Pull)

    async def go():
        task = asyncio.create_task(forwarder.run())
        for _ in range(50):
            if delivered:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())


def test_httpserver_forwards_the_packed_handshake(monkeypatch):
    status = PDUpKVStatus(9, 100, b"metadata")
    sent = []

    async def send(payload):
        sent.append(pickle.loads(payload))

    forwarder = PDMasterForwarder()
    event = asyncio.Event()
    forwarder.bind(group_req_id=8, event=event, websocket=SimpleNamespace(send=send))

    _run_one_ipc_message(monkeypatch, forwarder, status)
    assert event.websocket is not None
    assert sent == [(ObjType.PD_UPLOAD_KV_STATUS, status)]


def test_httpserver_drops_the_handshake_when_the_request_is_gone(monkeypatch):
    forwarder = PDMasterForwarder()
    status = PDUpKVStatus(8, 100, b"metadata")

    _run_one_ipc_message(monkeypatch, forwarder, status)


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
