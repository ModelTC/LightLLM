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
    return PD_Client_Obj(1, f"{mode}:8000", mode, {}, websocket=websocket)


def test_cancelled_caller_leaves_its_send_owned_by_the_connection():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        written = []

        async def send(payload):
            if payload == b"request":
                started.set()
                await release.wait()
            written.append(payload)

        node = _node(send_bytes=send)
        caller = asyncio.create_task(node.send_control_message(b"request"))
        await asyncio.wait_for(started.wait(), 1)
        owned_send = node._send_task
        assert owned_send is not caller
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not owned_send.cancelled()

        abort = asyncio.create_task(node.send_control_message(b"abort"))
        await asyncio.sleep(0)
        assert written == []
        release.set()
        await asyncio.wait_for(abort, 1)
        assert written == [b"request", b"abort"]
        assert node._send_task is None
        assert not node._send_lock.locked()

    asyncio.run(run())


def test_cancelled_queued_send_is_skipped_without_blocking_later_messages():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        written = []

        async def send(payload):
            if payload == b"first":
                started.set()
                await release.wait()
            written.append(payload)

        node = _node(send_bytes=send)
        first = asyncio.create_task(node.send_control_message(b"first"))
        await asyncio.wait_for(started.wait(), 1)
        cancelled = asyncio.create_task(node.send_control_message(b"cancelled"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        last = asyncio.create_task(node.send_control_message(b"last"))
        release.set()
        await asyncio.wait_for(asyncio.gather(first, last), 1)
        assert written == [b"first", b"last"]

    asyncio.run(run())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_failed_send_invalidates_connection_and_releases_waiters(cancel_caller):
    async def run():
        started, fail = asyncio.Event(), asyncio.Event()
        attempted = []

        async def send(payload):
            attempted.append(payload)
            started.set()
            await fail.wait()
            raise ConnectionError("peer disconnected")

        node = _node(send_bytes=send)
        first = asyncio.create_task(node.send_control_message(b"first"))
        await asyncio.wait_for(started.wait(), 1)
        owned_send = node._send_task
        if cancel_caller:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        queued = asyncio.create_task(node.send_control_message(b"queued"))
        await asyncio.sleep(0)
        fail.set()
        if not cancel_caller:
            with pytest.raises(ConnectionError, match="peer disconnected"):
                await asyncio.wait_for(first, 1)
        with pytest.raises(ConnectionError, match="unavailable"):
            await asyncio.wait_for(queued, 1)
        with pytest.raises(ConnectionError, match="peer disconnected"):
            await owned_send
        assert attempted == [b"first"]
        assert node.websocket is None
        assert node._send_task is None
        assert not node._send_lock.locked()

    asyncio.run(run())


def test_disconnect_rejects_new_and_queued_sends_while_an_old_send_is_pending():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        attempted = []

        async def send(payload):
            attempted.append(payload)
            started.set()
            await release.wait()

        node = _node(send_bytes=send)
        first = asyncio.create_task(node.send_control_message(b"first"))
        await asyncio.wait_for(started.wait(), 1)
        queued = asyncio.create_task(node.send_control_message(b"queued"))
        await asyncio.sleep(0)
        node.websocket = None
        with pytest.raises(ConnectionError, match="unavailable"):
            await asyncio.wait_for(node.send_control_message(b"new"), 0.1)
        release.set()
        await asyncio.wait_for(first, 1)
        with pytest.raises(ConnectionError, match="unavailable"):
            await asyncio.wait_for(queued, 1)
        assert attempted == [b"first"]
        assert not node._send_lock.locked()

    asyncio.run(run())


@pytest.mark.usefixtures("single_http_worker_start_args")
@pytest.mark.parametrize("connection_id", [None, "reused-id"])
def test_old_connection_cleanup_wakes_only_its_requests_and_keeps_replacement(connection_id):
    async def run():
        args = StartArgs()
        registry = PDManager(args)
        registration = dict(
            node_id=1, client_ip_port="p:8000", mode="prefill", start_args=vars(args), connection_id=connection_id
        )
        old = registry.register_pd(registration, SimpleNamespace(send_bytes=AsyncMock()))
        replacement = registry.register_pd(registration, SimpleNamespace(send_bytes=AsyncMock()))
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.pd_manager = registry
        old_status = ReqStatus(1, old, None)
        new_status = ReqStatus(2, replacement, None)
        manager.req_id_to_out_inf = {1: old_status, 2: new_status}

        await manager.remove_pd(old)
        assert old.websocket is None
        assert registry.url_to_pd_nodes[old.connection_key] is replacement
        assert old_status.event.is_set()
        assert old_status.prefill_prompt_ids_event.is_set()
        assert old_status.up_status_event.is_set()
        with pytest.raises(RuntimeError, match="disconnected"):
            old_status.raise_if_error()
        assert new_status.error_info is None
        assert not new_status.event.is_set()
        with pytest.raises(ConnectionError, match="unavailable"):
            await old.send_control_message(b"stale")
        await replacement.send_control_message(b"new")
        replacement.websocket.send_bytes.assert_awaited_once_with(b"new")

    asyncio.run(run())


def test_abort_notifies_the_other_peer_while_one_send_is_blocked():
    async def run():
        started, release, decode_notified = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def slow_send(payload):
            started.set()
            await release.wait()

        async def decode_send(payload):
            assert pickle.loads(payload) == (ObjType.ABORT, 123)
            decode_notified.set()

        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.req_id_to_out_inf = {123: ReqStatus(123, _node(send_bytes=slow_send), _node("decode", decode_send))}
        abort = asyncio.create_task(manager.abort(123))
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(decode_notified.wait(), 1)
        assert 123 not in manager.req_id_to_out_inf
        assert not abort.done()
        release.set()
        await asyncio.wait_for(abort, 1)

    asyncio.run(run())


def test_requests_handshake_and_abort_share_control_sender_and_preserve_int64_ids():
    async def run():
        manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        manager.args = SimpleNamespace(pd_node_id=1)
        manager.req_id_to_out_inf = {}
        p_node, d_node = _node(), _node("decode")
        for node in (p_node, d_node):
            node.send_control_message = AsyncMock(wraps=node.send_control_message)
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

        p_messages = [pickle.loads(call.args[0]) for call in p_node.websocket.send_bytes.await_args_list]
        d_messages = [pickle.loads(call.args[0]) for call in d_node.websocket.send_bytes.await_args_list]
        assert [message[0] for message in p_messages] == [ObjType.REQ, ObjType.PD_REQ_DECODE_NODE_INFO, ObjType.ABORT]
        assert [message[0] for message in d_messages] == [ObjType.REQ, ObjType.ABORT]
        assert d_messages[0][1][0].typecode == "q"
        assert d_messages[0][1][0] == ids
        assert p_node.send_control_message.await_count == 3
        assert d_node.send_control_message.await_count == 2

    asyncio.run(run())
