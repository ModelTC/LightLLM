import asyncio
import pickle
from array import array
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams, StartArgs
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, PDManager, ReqStatus
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj


def _node(mode="prefill", send_bytes=None):
    websocket = SimpleNamespace(send_bytes=send_bytes if send_bytes is not None else AsyncMock())
    node = PD_Client_Obj(1, f"{mode}:8000", mode, {})
    node.sub_connections["test"] = websocket
    return node


def _ws(node):
    return next(iter(node.sub_connections.values()))


@pytest.mark.usefixtures("single_http_worker_start_args")
@pytest.mark.parametrize("connection_id", [None, "reused-id"])
def test_old_connection_cleanup_wakes_only_its_requests_and_keeps_replacement(connection_id):
    async def run():
        args = StartArgs()
        registry = PDManager(args)
        registration = dict(
            node_id=1, client_ip_port="p:8000", mode="prefill", start_args=vars(args), connection_id=connection_id
        )
        old_ws = SimpleNamespace(send_bytes=AsyncMock())
        new_ws = SimpleNamespace(send_bytes=AsyncMock())
        node = registry.register_pd(registration, old_ws)
        registry.register_pd(registration, new_ws)
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.pd_manager = registry
        old_status = ReqStatus(req_id=1, p_node=node, d_node=None, p_websocket=old_ws, d_websocket=object())
        new_status = ReqStatus(req_id=2, p_node=node, d_node=None, p_websocket=new_ws, d_websocket=object())
        manager.req_id_to_out_inf = {1: old_status, 2: new_status}

        await manager.remove_pd(pd_client=node, websocket=old_ws)
        assert node.sub_connections == {connection_id: new_ws}
        assert old_status.event.is_set()
        assert old_status.prefill_prompt_ids_event.is_set()
        assert old_status.up_status_event.is_set()
        with pytest.raises(RuntimeError, match="disconnected"):
            old_status.raise_if_error()
        assert new_status.error_info is None
        assert not new_status.event.is_set()
        await new_ws.send_bytes(b"new")
        new_ws.send_bytes.assert_awaited_once_with(b"new")

    asyncio.run(run())


def test_direct_requests_handshake_and_abort_preserve_int64_ids():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(pd_node_id=1)
        manager.req_id_to_out_inf = {}
        p_node, d_node = _node(), _node("decode")
        params = SamplingParams()
        params.group_request_id, params.max_new_tokens = 123, 4
        ids = array("q", [1, 2 ** 40, 2 ** 63 - 1])
        decode_info = SimpleNamespace(ready_kv_len=len(ids) - 1)

        async def handshake(event, _request, **kwargs):
            if kwargs["stage"] == "prefill":
                event.prompt_ids = ids
            else:
                event.upkv_status = SimpleNamespace(pd_kv_trans_params=pickle.dumps(decode_info))
                status = manager.req_id_to_out_inf[123]
                status.out_token_info_list.append(
                    (
                        123,
                        "done",
                        {"count_output_tokens": 1, "node_mode": "decode"},
                        FinishStatus(FinishStatus.FINISHED_STOP),
                    )
                )
                status.event.set()

        manager._wait_for_event_or_disconnect = handshake
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        stream = manager.fetch_pd_stream(p_node, d_node, "prompt", params, MultimodalParams(), request)
        try:
            token = await asyncio.wait_for(anext(stream), 1)
            assert token[1] == "done"
        finally:
            await stream.aclose()
        await manager.abort(123)

        p_messages = [pickle.loads(call.args[0]) for call in _ws(p_node).send_bytes.await_args_list]
        d_messages = [pickle.loads(call.args[0]) for call in _ws(d_node).send_bytes.await_args_list]
        assert [message[0] for message in p_messages] == [ObjType.REQ, ObjType.PD_REQ_DECODE_NODE_INFO, ObjType.ABORT]
        assert [message[0] for message in d_messages] == [ObjType.REQ, ObjType.ABORT]
        assert d_messages[0][1][0].typecode == "q"
        assert d_messages[0][1][0] == ids
        assert _ws(p_node).send_bytes.await_count == 3
        assert _ws(d_node).send_bytes.await_count == 2

    asyncio.run(run())


def test_oversized_prefill_control_message_is_not_sent(monkeypatch):
    from lightllm.server.httpserver_for_pd_master import manager as manager_module
    from lightllm.utils.error_utils import InvalidRequestError

    monkeypatch.setattr(manager_module, "websocket_message_exceeds_limit", lambda payload: True)

    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(pd_node_id=1)
        manager.req_id_to_out_inf = {}
        p_node, d_node = _node(), _node("decode")
        params = SamplingParams()
        params.group_request_id, params.max_new_tokens = 7, 1
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        stream = manager.fetch_pd_stream(p_node, d_node, "prompt", params, MultimodalParams(), request)
        try:
            with pytest.raises(InvalidRequestError, match="websocket limit"):
                await anext(stream)
        finally:
            await stream.aclose()
        _ws(p_node).send_bytes.assert_not_awaited()
        _ws(d_node).send_bytes.assert_not_awaited()

    asyncio.run(run())


def test_oversized_decode_control_message_is_not_sent(monkeypatch):
    from lightllm.server.httpserver_for_pd_master import manager as manager_module
    from lightllm.utils.error_utils import InvalidRequestError

    calls = {"count": 0}

    def exceeds(payload):
        calls["count"] += 1
        return calls["count"] >= 2

    monkeypatch.setattr(manager_module, "websocket_message_exceeds_limit", exceeds)

    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(pd_node_id=1)
        manager.req_id_to_out_inf = {}
        p_node, d_node = _node(), _node("decode")
        params = SamplingParams()
        params.group_request_id, params.max_new_tokens = 7, 1

        async def wait(event, _request, **kwargs):
            event.prompt_ids = [1, 2]

        manager._wait_for_event_or_disconnect = wait
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        stream = manager.fetch_pd_stream(p_node, d_node, "prompt", params, MultimodalParams(), request)
        try:
            with pytest.raises(InvalidRequestError, match="websocket limit"):
                await anext(stream)
        finally:
            await stream.aclose()
        _ws(p_node).send_bytes.assert_awaited()
        _ws(d_node).send_bytes.assert_not_awaited()

    asyncio.run(run())
