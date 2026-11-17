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
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())


def _recording_websocket():
    sent = []

    async def send(payload):
        sent.append(pickle.loads(payload))

    return SimpleNamespace(send=send), sent


def test_httpserver_forwards_the_handshake_by_pd_master_node(monkeypatch):
    # 请求可能是从别的 httpserver worker 进来的，这里只靠 pd_master_node_id 选连接。
    status = PDUpKVStatus(9, 100, b"metadata")
    master_100, sent_100 = _recording_websocket()
    master_200, sent_200 = _recording_websocket()

    forwarder = PDMasterForwarder()
    forwarder.register(node_id=100, websocket=master_100)
    forwarder.register(node_id=200, websocket=master_200)

    _run_one_ipc_message(monkeypatch, forwarder, status)
    assert sent_100 == [(ObjType.PD_UPLOAD_KV_STATUS, status)]
    assert sent_200 == []


def test_httpserver_drops_the_handshake_without_master_connection(monkeypatch):
    forwarder = PDMasterForwarder()
    status = PDUpKVStatus(8, 100, b"metadata")

    _run_one_ipc_message(monkeypatch, forwarder, status)


def test_blocked_master_does_not_block_other_handshakes(monkeypatch):
    async def run():
        messages = asyncio.Queue()
        blocked = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()
        delivered = asyncio.Event()

        class Pull:
            def bind(self):
                pass

            async def recv(self):
                return await messages.get()

            def close(self):
                pass

        async def slow_send(payload):
            blocked.set()
            await release.wait()
            completed.set()

        async def healthy_send(payload):
            assert pickle.loads(payload) == (ObjType.PD_UPLOAD_KV_STATUS, PDUpKVStatus(16, 200, b"metadata"))
            delivered.set()

        monkeypatch.setattr("lightllm.server.pd_httpserver_ipc.PDHttpserverIpc.Pull", Pull)
        forwarder = PDMasterForwarder()
        forwarder.register(100, SimpleNamespace(send=slow_send))
        forwarder.register(200, SimpleNamespace(send=healthy_send))
        task = asyncio.create_task(forwarder.run())
        try:
            await messages.put(PDUpKVStatus(8, 100, b"metadata"))
            await asyncio.wait_for(blocked.wait(), timeout=1)
            await messages.put(PDUpKVStatus(16, 200, b"metadata"))
            await asyncio.wait_for(delivered.wait(), timeout=1)
            assert not completed.is_set()
        finally:
            release.set()
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)
            await asyncio.wait_for(completed.wait(), timeout=1)

    asyncio.run(run())


def test_stale_connection_does_not_unregister_its_replacement():
    old, _ = _recording_websocket()
    new, _ = _recording_websocket()
    forwarder = PDMasterForwarder()
    forwarder.register(node_id=100, websocket=old)
    forwarder.register(node_id=100, websocket=new)

    forwarder.unregister(node_id=100, websocket=old)
    assert forwarder._node_id_to_websocket == {100: new}

    forwarder.unregister(node_id=100, websocket=new)
    assert forwarder._node_id_to_websocket == {}


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
