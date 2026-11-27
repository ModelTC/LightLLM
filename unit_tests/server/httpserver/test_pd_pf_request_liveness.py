"""PD waits abort the current segment when its original choice is inactive."""

from builtins import anext
import asyncio
import pickle
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams, StartArgs
from lightllm.server.httpserver.pd_pf_prefill_node import run_pd_pf_prefill_request
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, ReqStatus
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj, PDDecodeNodeInfo, PDUpKVStatus, PdPfPrefillKvReady
from lightllm.utils.error_utils import ClientDisconnected, RequestStopped, ServerBusyError


GROUP_ID = 160
ORIGIN_ID = 80
SIBLING_ID = 81


def _token(text, count, finish=FinishStatus.NO_FINISH, node_mode=None):
    metadata = {"id": count, "count_output_tokens": count}
    if node_mode is not None:
        metadata["node_mode"] = node_mode
    return GROUP_ID, text, metadata, FinishStatus(finish)


class _Session:
    def __init__(self, tokens=(), prefill_first=True, wrap_default=True):
        self.manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
        self.manager.args = StartArgs(pd_node_id=1, pd_prefill_first=prefill_first)
        self.manager.req_id_to_out_inf = {}
        self.manager.active_req_ids = {ORIGIN_ID, SIBLING_ID}
        self.manager.pd_manager = SimpleNamespace(selector=SimpleNamespace(selection_lock=asyncio.Lock()))
        self.manager.abort = AsyncMock(wraps=self.manager.abort)
        self.params = SamplingParams()
        self.params.init(None, max_new_tokens=4)
        self.params.group_request_id = GROUP_ID
        self.tokens = tokens
        self.hold_handshake = False
        self.hold_prefill_ready = False
        self.failed_sends = set()
        self.messages = {"p": [], "d": []}
        self.request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        self.p_ws = SimpleNamespace(send_bytes=AsyncMock(side_effect=self._send_prefill))
        self.d_ws = SimpleNamespace(send_bytes=AsyncMock(side_effect=self._send_decode))
        self.p_node = PD_Client_Obj(1, "prefill:8000", "prefill", {}, sub_connections={"p": self.p_ws})
        self.d_node = PD_Client_Obj(2, "decode:8000", "decode", {}, sub_connections={"d": self.d_ws})
        self.manager.select_p_d_node = AsyncMock(return_value=(self.p_node, self.d_node, None))
        if prefill_first:
            self.stream = self.manager.fetch_prefill_first_stream(
                self.p_node, "prompt", self.params, MultimodalParams(), self.request, ORIGIN_ID
            )
        elif wrap_default:
            self.stream = self.manager._wait_to_token_package(
                self.p_node,
                self.d_node,
                time.time(),
                "prompt",
                self.params,
                MultimodalParams(),
                self.request,
                ORIGIN_ID,
            )
        else:
            self.stream = self.manager.fetch_pd_stream(
                p_node=self.p_node,
                d_node=self.d_node,
                prompt="prompt",
                sampling_params=self.params,
                multimodal_params=MultimodalParams(),
                request=self.request,
                origin_request_id=ORIGIN_ID,
            )

    def _record(self, side, payload):
        message = pickle.loads(payload)
        self.messages[side].append(message)
        if message[0] == ObjType.ABORT and side in self.failed_sends:
            raise ConnectionError(f"{side} connection closed")
        return message

    async def _send_prefill(self, payload):
        message = self._record("p", payload)
        if message[0] == ObjType.PD_PF_PREFILL_REQ:
            self.status = self.manager.req_id_to_out_inf[GROUP_ID]
            self.status.pf_prefill_kv_ready_event.ready = PdPfPrefillKvReady(GROUP_ID, [1, 2])
            if not self.hold_prefill_ready:
                self.status.pf_prefill_kv_ready_event.set()
            self.status.out_token_info_list.append(
                _token("hello", 1, FinishStatus.FINISHED_LENGTH, node_mode="prefill")
            )
            self.status.event.set()
        elif message[0] == ObjType.REQ:
            self.status = self.manager.req_id_to_out_inf[GROUP_ID]
            self.status.prefill_prompt_ids_event.prompt_ids = [1, 2]
            if not self.hold_prefill_ready:
                self.status.prefill_prompt_ids_event.set()

    async def _send_decode(self, payload):
        message = self._record("d", payload)
        if message[0] == ObjType.REQ:
            ready_kv_len = 0 if self.manager.args.pd_prefill_first else 1
            info = PDDecodeNodeInfo(2, 1, "decode", b"metadata", 1, b"pages", GROUP_ID, ready_kv_len)
            if not self.hold_handshake:
                await self.manager.update_req_status(PDUpKVStatus(GROUP_ID, 1, pickle.dumps(info)))
            self.status.out_token_info_list.extend(self.tokens)
            if self.tokens:
                self.status.event.set()

    def assert_cleaned(self, has_decode=True):
        assert GROUP_ID not in self.manager.req_id_to_out_inf
        assert SIBLING_ID in self.manager.active_req_ids
        self.manager.abort.assert_awaited_once_with(
            group_request_id=GROUP_ID, p_node=self.p_node, d_node=self.d_node if has_decode else None
        )
        for side in ("p", "d"):
            expected = [(ObjType.ABORT, GROUP_ID)] if side == "p" or has_decode else []
            assert [message for message in self.messages[side] if message[0] == ObjType.ABORT] == expected


@pytest.mark.parametrize(
    "prefill_first,stage", [(True, "before_handshake"), (True, "waiting_for_tokens"), (False, "waiting_for_tokens")]
)
@pytest.mark.parametrize(
    "scenario",
    [
        "stopped",
        "record_removed",
        "buffered_tokens",
        "stale_error",
        "p_send_error",
        "d_send_error",
        "both_send_error",
        "sibling",
    ],
)
def test_inactive_choice_aborts_and_returns(prefill_first, stage, scenario):
    async def run():
        session = _Session(
            tokens=() if prefill_first else [_token("hello", 1)], prefill_first=prefill_first, wrap_default=False
        )
        session.hold_handshake = stage == "before_handshake"
        if scenario in ("p_send_error", "both_send_error"):
            session.failed_sends.add("p")
        if scenario in ("d_send_error", "both_send_error"):
            session.failed_sends.add("d")
        sibling = ReqStatus(168, None, None, None, None)
        if scenario == "sibling":
            session.manager.req_id_to_out_inf[168] = sibling
        waiting, release = asyncio.Event(), asyncio.Event()
        pending = None
        try:
            assert (await anext(session.stream))[1] == "hello"
            if stage == "waiting_for_tokens":

                async def wait():
                    waiting.set()
                    await release.wait()

                session.status.wait_to_ready = wait
                pending = asyncio.create_task(anext(session.stream))
                await waiting.wait()

            await session.manager.stop_request(ORIGIN_ID)
            assert GROUP_ID in session.manager.req_id_to_out_inf
            if scenario == "record_removed":
                await session.manager.remove_req(GROUP_ID)
            if scenario == "buffered_tokens":
                session.status.out_token_info_list.append(_token("late", 2, FinishStatus.FINISHED_STOP))
            elif scenario == "stale_error":
                session.status.error_info = "late error for a stopped request"
            release.set()
            with pytest.raises(StopAsyncIteration):
                if pending is None:
                    await anext(session.stream)
                else:
                    await pending
            session.assert_cleaned()
            if scenario == "sibling":
                assert session.manager.req_id_to_out_inf[168] is sibling
            if stage == "before_handshake":
                assert not session.status.up_status_event.is_set()
                assert not any(message[0] == ObjType.PD_PF_DECODE_KV_READY for message in session.messages["p"])
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await session.stream.aclose()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize("prefill_first,wrap_default", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("stage", ["prefill", "decode"])
@pytest.mark.parametrize("wake", ["silent", "ready", "error"])
def test_stop_during_handshake_aborts_only_current_choice(prefill_first, wrap_default, stage, wake):
    async def run():
        session = _Session(prefill_first=prefill_first, wrap_default=wrap_default)
        session.hold_handshake = True
        session.hold_prefill_ready = stage == "prefill"
        target_stage = (
            {"prefill": "pf-prefill-kv-ready", "decode": "prefill-first-decode"}[stage] if prefill_first else stage
        )
        waiting = asyncio.Event()
        handshake_event = None
        original_wait = session.manager._wait_for_event_or_disconnect

        async def wait_for_event(event, request, timeout, group_request_id, origin_request_id, stage):
            nonlocal handshake_event
            assert origin_request_id == ORIGIN_ID
            if stage == target_stage:
                handshake_event = event
                event_wait = event.wait

                async def observed_wait():
                    waiting.set()
                    await event_wait()

                event.wait = observed_wait
            await original_wait(event, request, timeout, group_request_id, origin_request_id, stage)

        session.manager._wait_for_event_or_disconnect = wait_for_event
        pending = None
        try:
            if prefill_first and stage == "decode":
                assert (await anext(session.stream))[1] == "hello"
            pending = asyncio.create_task(anext(session.stream))
            await waiting.wait()
            await session.manager.stop_request(ORIGIN_ID)
            if wake == "ready":
                handshake_event.set()
            elif wake == "error":
                await session.status.set_error("late error for stopped choice")
            with pytest.raises(StopAsyncIteration):
                await pending
            session.assert_cleaned(has_decode=not prefill_first or stage == "decode")
            assert not any(
                message[0] in (ObjType.PD_PF_DECODE_KV_READY, ObjType.PD_REQ_DECODE_NODE_INFO)
                for message in session.messages["p"]
            )
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await session.stream.aclose()

    asyncio.run(asyncio.wait_for(run(), timeout=3))


@pytest.mark.parametrize("stage", ["prefill", "decode", "pf-prefill-kv-ready", "prefill-first-decode"])
@pytest.mark.parametrize("ready", [False, True])
def test_wait_raises_request_stopped_for_original_id(stage, ready):
    async def run():
        session = _Session()
        await session.manager.stop_request(ORIGIN_ID)
        event = asyncio.Event()
        if ready:
            event.set()
        with pytest.raises(RequestStopped) as error:
            await session.manager._wait_for_event_or_disconnect(event, session.request, 180, GROUP_ID, ORIGIN_ID, stage)
        assert error.value.origin_request_id == ORIGIN_ID
        session.manager.abort.assert_not_awaited()
        assert SIBLING_ID in session.manager.active_req_ids

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize(
    "finish", [FinishStatus.FINISHED_STOP, FinishStatus.FINISHED_LENGTH, FinishStatus.FINISHED_PD_DECODE_CAPACITY]
)
def test_existing_segment_preserves_output_and_finish_status(finish):
    async def run():
        count = 1 if finish == FinishStatus.FINISHED_PD_DECODE_CAPACITY else 2
        session = _Session(tokens=[_token("hello", 1), _token(" world", count, finish)])
        try:
            first = await anext(session.stream)
            assert first[1] == "hello"
            assert not first[3].is_finished()
            last = await anext(session.stream)
            assert last[1] == " world"
            assert last[3].status == finish
            session.manager.abort.assert_not_awaited()
            assert ORIGIN_ID in session.manager.active_req_ids
            assert GROUP_ID not in session.manager.active_req_ids
        finally:
            await session.stream.aclose()
        session.assert_cleaned()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize("failure", ["node_error", "node_busy", "client_disconnect", "cancel"])
def test_existing_segment_preserves_failure_cleanup(failure):
    async def run():
        session = _Session()
        pending = None
        try:
            await anext(session.stream)
            if failure in ("node_error", "node_busy"):
                await session.status.set_error("decode failed", is_server_busy=failure == "node_busy")
                expected = ServerBusyError if failure == "node_busy" else RuntimeError
                with pytest.raises(expected, match="decode failed"):
                    await anext(session.stream)
            elif failure == "client_disconnect":
                session.request.is_disconnected.return_value = True
                with pytest.raises(ClientDisconnected):
                    await anext(session.stream)
            else:
                waiting = asyncio.Event()
                wait_to_ready = session.status.wait_to_ready

                async def wait():
                    waiting.set()
                    await wait_to_ready()

                session.status.wait_to_ready = wait
                pending = asyncio.create_task(anext(session.stream))
                await waiting.wait()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            session.assert_cleaned()
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await session.stream.aclose()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_decode_is_reserved_before_the_prefill_token_arrives():
    async def run():
        session = _Session()
        ready = asyncio.Event()
        decode = asyncio.Event()
        send_prefill = session.p_ws.send_bytes.side_effect
        send_decode = session.d_ws.send_bytes.side_effect

        async def hold_token(payload):
            await send_prefill(payload)
            if pickle.loads(payload)[0] == ObjType.PD_PF_PREFILL_REQ:
                session.status.out_token_info_list.clear()
                session.status.event.clear()
                ready.set()

        async def on_decode(payload):
            await send_decode(payload)
            if pickle.loads(payload)[0] == ObjType.REQ:
                decode.set()

        session.p_ws.send_bytes.side_effect = hold_token
        session.d_ws.send_bytes.side_effect = on_decode
        pending = asyncio.create_task(anext(session.stream))
        await asyncio.wait_for(ready.wait(), timeout=2)
        await asyncio.wait_for(decode.wait(), timeout=2)
        assert not pending.done()

        async def endpoint_sent():
            while not any(m[0] == ObjType.PD_PF_DECODE_KV_READY for m in session.messages["p"]):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(endpoint_sent(), timeout=2)
        assert not pending.done()
        session.status.out_token_info_list.append(_token("first", 1, FinishStatus.FINISHED_LENGTH, node_mode="prefill"))
        session.status.event.set()
        assert (await asyncio.wait_for(pending, timeout=2))[1] == "first"
        assert decode.is_set()
        await session.stream.aclose()
        session.assert_cleaned()

    asyncio.run(run())


def test_prefill_token_is_taken_when_decode_output_is_ahead():
    async def run():
        session = _Session()
        send_prefill = session.p_ws.send_bytes.side_effect

        async def put_decode_token_first(payload):
            await send_prefill(payload)
            if pickle.loads(payload)[0] == ObjType.PD_PF_PREFILL_REQ:
                session.status.out_token_info_list.insert(0, _token("skip", 1, node_mode="decode"))

        session.p_ws.send_bytes.side_effect = put_decode_token_first
        first = await anext(session.stream)
        assert first[1] == "hello"
        assert not first[3].is_finished()
        assert [item[1] for item in session.status.out_token_info_list] == ["skip"]
        await session.stream.aclose()
        session.assert_cleaned()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


@pytest.mark.parametrize("scheduled", [True, False])
def test_last_chunk_reports_prompt_once_and_transfers_only_after_prefill(scheduled):
    async def run():
        params = SamplingParams()
        params.init(None, max_new_tokens=1)
        params.group_request_id = GROUP_ID
        reported = asyncio.Event()
        transfer = Mock()
        shm_req = SimpleNamespace(
            shm_prompt_ids=SimpleNamespace(arr=[11, 12, 0]),
            input_len=2,
            pd_last_chunk_scheduled=False,
            sample_params=SimpleNamespace(pd_kv_trans_params=SimpleNamespace(set=transfer)),
        )
        requests = {GROUP_ID: SimpleNamespace(group_req_objs=SimpleNamespace(shm_req_objs=[shm_req]))}
        transfer.side_effect = lambda payload: requests.pop(GROUP_ID)
        decode_ready = asyncio.Event()
        decode_ready.decode_node_info = SimpleNamespace(ready_kv_len=0)
        decode_ready.set()

        async def generate(**kwargs):
            if scheduled:
                shm_req.pd_last_chunk_scheduled = True
                await reported.wait()
                transfer.assert_not_called()
            yield _token("first", 1, FinishStatus.FINISHED_LENGTH)

        async def send(payload):
            kind, ready = pickle.loads(payload)
            assert kind == ObjType.PD_PF_PREFILL_KV_READY
            assert ready.group_request_id == GROUP_ID and ready.prompt_ids == [11, 12]
            reported.set()

        manager = SimpleNamespace(
            args=SimpleNamespace(run_mode="prefill"), req_id_to_out_inf=requests, generate=generate
        )
        ws = SimpleNamespace(send=AsyncMock(side_effect=send))
        await run_pd_pf_prefill_request(
            manager, "prompt", params, MultimodalParams(), asyncio.Queue(), ws, decode_ready
        )
        ws.send.assert_awaited_once()
        transfer.assert_called_once()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_last_chunk_single_output_never_submits_decode():
    async def run():
        session = _Session()
        session.params.max_new_tokens = 1
        assert (await anext(session.stream))[3].is_finished_length()
        assert session.messages["p"][0][1][1].pd_prefill_first
        assert not session.messages["d"]
        await session.stream.aclose()
        session.assert_cleaned(has_decode=False)

    asyncio.run(run())


def test_stop_during_last_chunk_releases_decode_reservation():
    async def run():
        session = _Session()
        original_send = session.d_ws.send_bytes.side_effect

        async def stop_before_first_token(payload):
            await original_send(payload)
            if pickle.loads(payload)[0] == ObjType.REQ:
                session.status.out_token_info_list.clear()
                await session.manager.stop_request(ORIGIN_ID)

        session.d_ws.send_bytes.side_effect = stop_before_first_token
        with pytest.raises(StopAsyncIteration):
            await anext(session.stream)
        session.assert_cleaned()
        assert not any(m[0] == ObjType.PD_PF_DECODE_KV_READY for m in session.messages["p"])

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def _hold_token_until_decode(session):
    ready = asyncio.Event()
    decode = asyncio.Event()
    send_prefill = session.p_ws.send_bytes.side_effect
    send_decode = session.d_ws.send_bytes.side_effect

    async def hold_token(payload):
        await send_prefill(payload)
        if pickle.loads(payload)[0] == ObjType.PD_PF_PREFILL_REQ:
            session.status.out_token_info_list.clear()
            session.status.event.clear()
            ready.set()

    async def on_decode(payload):
        await send_decode(payload)
        if pickle.loads(payload)[0] == ObjType.REQ:
            decode.set()

    session.p_ws.send_bytes.side_effect = hold_token
    session.d_ws.send_bytes.side_effect = on_decode
    return ready, decode


def test_finished_first_token_after_reservation_skips_transfer():
    async def run():
        session = _Session()
        _, decode = _hold_token_until_decode(session)
        pending = asyncio.create_task(anext(session.stream))
        await asyncio.wait_for(decode.wait(), timeout=2)
        session.status.out_token_info_list.append(_token("end", 1, FinishStatus.FINISHED_STOP, node_mode="prefill"))
        session.status.event.set()
        item = await asyncio.wait_for(pending, timeout=2)
        assert item[1] == "end"
        assert item[3].is_finished()
        await session.stream.aclose()
        session.assert_cleaned()

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_disconnect_after_reservation_aborts_decode():
    async def run():
        session = _Session()
        _, decode = _hold_token_until_decode(session)
        pending = asyncio.create_task(anext(session.stream))
        await asyncio.wait_for(decode.wait(), timeout=2)
        session.request.is_disconnected = AsyncMock(return_value=True)
        session.status.event.set()
        with pytest.raises(ClientDisconnected):
            await pending
        session.assert_cleaned()

    asyncio.run(asyncio.wait_for(run(), timeout=2))
