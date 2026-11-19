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

    packed = pickle.loads(sent[0])
    assert packed[0] == 100
    status = packed[1]
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


def test_pd_report_routes_the_handshake_by_master_id():
    manager = PdReportManager(args=SimpleNamespace(), pull=None)
    manager._queues[100] = asyncio.Queue()
    manager._queues[200] = asyncio.Queue()
    status = PDUpKVStatus(9, 100, b"metadata")

    manager.forward_report(message=(100, status))

    assert manager._queues[100].get_nowait() == (ObjType.PD_UPLOAD_KV_STATUS, status)
    assert manager._queues[200].empty()


def test_pd_report_drops_a_handshake_for_an_unknown_master():
    manager = PdReportManager(args=SimpleNamespace(), pull=None)
    manager._queues[100] = asyncio.Queue()
    manager.forward_report(message=(100, PDUpKVStatus(8, 200, b"metadata")))
    manager.forward_report(message=(300, PDUpKVStatus(8, 300, b"metadata")))
    assert manager._queues[100].empty()
    assert 300 not in manager._queues


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
