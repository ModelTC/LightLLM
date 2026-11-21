"""PD waits abort the current segment when its original choice is inactive."""

import asyncio
import pickle
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams, StartArgs
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster, ReqStatus
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj, PDDecodeNodeInfo, PDUpKVStatus, PdPfPrefillKvReady
from lightllm.utils.error_utils import ClientDisconnected, RequestStopped, ServerBusyError


GROUP_ID = 160
ORIGIN_ID = 80
SIBLING_ID = 81


def _token(text, count, finish=FinishStatus.NO_FINISH):
    return GROUP_ID, text, {"id": count, "count_output_tokens": count}, FinishStatus(finish)


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
            self.status.out_token_info_list.append(_token("hello", 1, FinishStatus.FINISHED_LENGTH))
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
