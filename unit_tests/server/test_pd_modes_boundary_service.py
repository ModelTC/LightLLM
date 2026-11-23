"""Run the same HTTP/WebSocket boundary scenarios against both PD master modes.

API workers, HAProxy and metrics processes are real. Protocol peers replace GPU
inference so waits, ordering, failures and cancellation can be controlled.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import pickle

import httpx
import pytest

from lightllm.server.core.objs import FinishStatus
from lightllm.server.pd_io_struct import ObjType
from unit_tests.server.pd_master_test_utils import running_pd_master
from unit_tests.server.test_pd_prefill_first_service import (
    PrefillFirstPeers,
    Segment,
    _capacity_segments,
    _generate,
    _params_view,
    _token_message,
    _wait_until,
)


@dataclass
class BoundarySegment(Segment):
    decode_busy: bool = False
    close_decode: bool = False
    error_before_upkv: str = ""


class BoundaryPeers(PrefillFirstPeers):
    def __init__(self, master, prefill_first, segments):
        super().__init__(master, segments=segments)
        self.prefill_first = prefill_first

    async def _run(self, mode, master_id, node_name, websocket):
        async for data in websocket:
            message = pickle.loads(data)
            kind = message[0]
            self.message_types.append((mode, master_id, kind))
            if kind == ObjType.ABORT:
                self.aborts.append((mode, master_id, message[1]))
            elif mode == "prefill" and kind in (ObjType.REQ, ObjType.PD_PF_PREFILL_REQ):
                expected = ObjType.PD_PF_PREFILL_REQ if self.prefill_first else ObjType.REQ
                assert kind == expected
                self._track(self._reply_prefill(websocket, master_id, node_name, message))
            elif mode == "prefill" and kind in (ObjType.PD_REQ_DECODE_NODE_INFO, ObjType.PD_PF_DECODE_KV_READY):
                _, request_id, info = message
                self.kv_ready.append((master_id, request_id, info))
                if not self.prefill_first:
                    segment = self._group_segments[request_id]
                    if info.ready_kv_len != len(self._prompt_ids(request_id)) - 1:
                        finish = segment.prefill_finish
                        if finish == FinishStatus.NO_FINISH:
                            finish = FinishStatus.FINISHED_LENGTH
                        await self._send(
                            websocket,
                            _token_message(
                                "prefill",
                                master_id,
                                request_id,
                                segment.prefill_text,
                                finish,
                                1,
                                segment.prefill_cache_len,
                            ),
                        )
            elif mode == "decode" and kind == ObjType.REQ:
                self._track(self._reply_decode(websocket, master_id, node_name, message))

    def _prompt_ids(self, request_id):
        return next(req["prompt_ids"] for req in self.prefill_reqs if req["params"]["group_request_id"] == request_id)

    async def _reply_prefill(self, websocket, master_id, node_name, message):
        if self.prefill_first:
            await super()._reply_prefill(websocket, master_id, node_name, message)
            return
        _, (prompt, params, _) = message
        view = _params_view(params)
        request_id = view["group_request_id"]
        segment = self._segment_for_prefill(master_id, request_id)
        prompt_ids = segment.prompt_ids if segment.prompt_ids is not None else [len(prompt), 2, 3]
        self.prefill_reqs.append(
            dict(master_id=master_id, node_name=node_name, prompt=prompt, prompt_ids=list(prompt_ids), params=view)
        )
        if segment.close_prefill:
            await websocket.close()
            return
        if segment.hold_prefill:
            await self.release_prefill.wait()
        if segment.busy:
            await self._send(websocket, (ObjType.PD_UPLOAD_SERVER_BUSY, request_id, "test capacity exhausted"))
        elif segment.prefill_error:
            await self._send(websocket, (ObjType.PD_UPLOAD_GENERATE_ERROR, request_id, segment.prefill_error))
        else:
            await self._send(websocket, (ObjType.PD_UPLOAD_PREFILL_PROMPT_IDS, request_id, prompt_ids))

    async def _reply_decode(self, websocket, master_id, node_name, message):
        _, (prompt_ids, params, _) = message
        segment = self._group_segments[params.group_request_id]
        if (
            getattr(segment, "decode_busy", False)
            or getattr(segment, "close_decode", False)
            or getattr(segment, "error_before_upkv", "")
        ):
            self.decode_reqs.append(
                dict(master_id=master_id, node_name=node_name, prompt_ids=list(prompt_ids), params=_params_view(params))
            )
            if segment.close_decode:
                await websocket.close()
            elif segment.decode_busy:
                await self._send(websocket, (ObjType.PD_UPLOAD_SERVER_BUSY, params.group_request_id, "decode busy"))
            else:
                await self._send(
                    websocket, (ObjType.PD_UPLOAD_GENERATE_ERROR, params.group_request_id, segment.error_before_upkv)
                )
            return
        if not self.prefill_first and params.max_new_tokens == 1:
            segment.decode_steps = ((segment.prefill_text, FinishStatus.FINISHED_LENGTH, 1),)
        await super()._reply_decode(websocket, master_id, node_name, message)


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "prefill_first"])
def pd_service(request, tmp_path_factory):
    mode = request.param
    args = ["--httpserver_workers", "2", "--select_p_d_node_strategy", "round_robin"]
    if mode:
        args.append("--pd_prefill_first")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS", "0")
        with running_pd_master(tmp_path_factory.mktemp("pd-boundaries"), extra_args=args) as master:
            assert len(master.workers) == 2
            yield master, mode


@asynccontextmanager
async def _open(service, segments=None):
    master, mode = service
    peers = BoundaryPeers(master, mode, segments or [BoundarySegment()])
    try:
        await peers.connect()
        async with httpx.AsyncClient(trust_env=False, timeout=12) as client:
            yield peers, client
        assert not peers.peer_errors, peers.peer_errors
    finally:
        await peers.close()


def _run(service, scenario):
    try:
        asyncio.run(asyncio.wait_for(scenario(), timeout=25))
    except BaseException:
        print(service[0].log_path.read_text(errors="replace")[-12000:])
        raise


async def _assert_aborted(peers, request_id, sides=("prefill", "decode")):
    await _wait_until(
        lambda: all(any(item[0] == side and item[2] == request_id for item in peers.aborts) for side in sides),
        timeout=8,
    )


@pytest.mark.parametrize(
    "case",
    [
        "length_one",
        "length_two",
        "prefill_length",
        "prefill_stop",
        "cache_zero",
        "cache_full",
        "large_ids",
        "ready_before_token",
        "error_terminal_index_one",
        "abort_terminal_index_one",
        "duplicate_first",
        "unmatched_stop",
        "three_choices",
    ],
)
def test_response_boundaries(pd_service, case):
    async def scenario():
        segment = BoundarySegment()
        maximum, expected, reason, extra = 4, "hello world", "stop", {}
        if case == "length_one":
            maximum, expected, reason = 1, "hello", "length"
            segment.prefill_finish = FinishStatus.FINISHED_LENGTH
        elif case == "length_two":
            maximum, reason = 2, "length"
            segment.decode_steps = (("hello", FinishStatus.NO_FINISH, 1), (" world", FinishStatus.FINISHED_LENGTH, 2))
        elif case == "prefill_length":
            segment.prefill_finish = FinishStatus.FINISHED_LENGTH
        elif case == "prefill_stop":
            segment.prefill_finish, expected = FinishStatus.FINISHED_STOP, "hello"
        elif case == "cache_full":
            segment.prompt_ids, segment.ready_kv_len = [2, 3, 2], 2
        elif case == "large_ids":
            segment.prompt_ids = [1, 2 ** 40, 2 ** 63 - 1]
        elif case == "ready_before_token":
            segment.token_before_kv_ready = False
        elif case in ("error_terminal_index_one", "abort_terminal_index_one"):
            finish = FinishStatus.FINISHED_ERROR if case.startswith("error") else FinishStatus.FINISHED_ABORTED
            segment.decode_steps = (("hello", FinishStatus.NO_FINISH, 1), ("", finish, 1))
            expected, reason = "hello", "error" if case.startswith("error") else "abort"
        elif case == "unmatched_stop":
            extra["stop_sequences"] = ["absent"]
        elif case == "three_choices":
            extra.update(n=3, best_of=3)
        async with _open(pd_service, [segment]) as (peers, client):
            response = await _generate(client, pd_service[0].public_url, max_new_tokens=maximum, extra=extra)
            assert response.status_code == 200, response.text
            body = response.json()
            count = 3 if case == "three_choices" else 1
            assert body["generated_text"] == [expected] * count, body
            assert body["finish_reason"] == ([reason] * count if count > 1 else reason), body
            assert len({req["params"]["group_request_id"] for req in peers.prefill_reqs}) == count
            if case == "large_ids":
                assert peers.decode_reqs[0]["prompt_ids"] == segment.prompt_ids
            if pd_service[1] and case in ("length_one", "prefill_stop"):
                assert not peers.decode_reqs

    _run(pd_service, scenario)


@pytest.mark.parametrize(
    "case",
    [
        "prefill_busy",
        "prefill_error",
        "decode_busy",
        "decode_error_before",
        "decode_error_after",
        "prefill_disconnect",
        "decode_disconnect",
    ],
)
def test_protocol_failures(pd_service, case):
    async def scenario():
        segment = BoundarySegment()
        if case == "prefill_busy":
            segment.busy = True
        elif case == "prefill_error":
            segment.prefill_error = "boundary prefill failed"
        elif case == "decode_busy":
            segment.decode_busy = True
        elif case == "decode_error_before":
            segment.error_before_upkv = "boundary decode failed"
        elif case == "decode_error_after":
            segment.decode_error = "boundary transfer failed"
        elif case == "prefill_disconnect":
            segment.close_prefill = True
        else:
            segment.close_decode = True
        async with _open(pd_service, [segment]) as (peers, client):
            response = await _generate(client, pd_service[0].public_url)
            assert response.status_code == (429 if case.endswith("busy") else 417), response.text
            if case.startswith("prefill_"):
                assert not peers.decode_reqs
            elif case != "decode_disconnect":
                await _assert_aborted(peers, peers.prefill_reqs[0]["params"]["group_request_id"])

    _run(pd_service, scenario)


@pytest.mark.parametrize("case", ["string_first", "token_id_first", "include_string", "split_string"])
def test_application_stop_releases_silent_nodes(pd_service, case):
    async def scenario():
        segment = BoundarySegment(decode_steps=(), hold_upkv=pd_service[1] and case != "split_string")
        extra, expected = {"stop_sequences": ["hello"]}, ""
        if case == "token_id_first":
            extra["stop_sequences"] = [[2]]
            expected = "hello"
        elif case == "include_string":
            extra["include_stop_str_in_output"], expected = True, "hello"
        elif case == "split_string":
            extra["stop_sequences"] = ["hello world"]
            segment.decode_steps = (("hello", FinishStatus.NO_FINISH, 1), (" world", FinishStatus.NO_FINISH, 2))
        async with _open(pd_service, [segment]) as (peers, client):
            response = await _generate(client, pd_service[0].public_url, extra=extra)
            assert response.status_code == 200, response.text
            assert response.json()["generated_text"] == [expected], response.text
            assert response.json()["finish_reason"] == "stop", response.text
            await _assert_aborted(peers, peers.prefill_reqs[0]["params"]["group_request_id"])

    _run(pd_service, scenario)


@pytest.mark.parametrize("stage", ["prefill", "handshake", "decode"])
def test_disconnect_cancels_current_segment(pd_service, stage):
    async def scenario():
        segment = BoundarySegment(hold_prefill=stage == "prefill", hold_upkv=stage == "handshake", decode_steps=())
        async with _open(pd_service, [segment]) as (peers, client):
            if stage == "decode":
                async with client.stream(
                    "POST",
                    pd_service[0].public_url + "/generate_stream",
                    json={"inputs": "hello", "parameters": {"max_new_tokens": 4}},
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            assert json.loads(line[5:])["token"]["text"] == "hello"
                            break
            else:
                pending = asyncio.create_task(_generate(client, pd_service[0].public_url))
                await _wait_until(lambda: bool(peers.prefill_reqs if stage == "prefill" else peers.decode_reqs))
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            request_id = peers.prefill_reqs[0]["params"]["group_request_id"]
            await _assert_aborted(
                peers, request_id, sides=("prefill",) if stage == "prefill" else ("prefill", "decode")
            )
            if stage == "prefill":
                assert not peers.decode_reqs

    _run(pd_service, scenario)


def test_capacity_continuation_preserves_output_and_priority(pd_service):
    async def scenario():
        async with _open(pd_service, _capacity_segments()) as (peers, client):
            response = await _generate(client, pd_service[0].public_url, max_new_tokens=3)
            assert response.status_code == 200, response.text
            assert response.json()["generated_text"] == ["ACD"], response.text
            assert [req["prompt"] for req in peers.prefill_reqs] == ["hello", "helloA"]
            assert [req["params"]["infer_high_priority"] for req in peers.prefill_reqs] == [0, -1]

    _run(pd_service, scenario)


def test_stopped_choice_keeps_sibling_running(pd_service):
    async def scenario():
        segments = [
            BoundarySegment(hold_upkv=pd_service[1], decode_steps=()),
            BoundarySegment(
                prefill_text="world",
                decode_steps=(("world", FinishStatus.NO_FINISH, 1), (" world", FinishStatus.FINISHED_STOP, 2)),
            ),
        ]
        async with _open(pd_service, segments) as (peers, client):
            url = "http://" + pd_service[0].workers[0]["host_ip_port"]
            response = await _generate(client, url, extra={"n": 2, "best_of": 2, "stop_sequences": ["hello"]})
            assert response.status_code == 200, response.text
            assert sorted(response.json()["generated_text"]) == ["", "world world"], response.text
            stopped_id = peers.prefill_reqs[0]["params"]["group_request_id"]
            await _assert_aborted(peers, stopped_id)
            assert len(peers.prefill_reqs) == 2

    _run(pd_service, scenario)


def test_concurrent_request_ids_and_worker_affinity(pd_service):
    async def scenario():
        async with _open(pd_service) as (peers, client):
            responses = await asyncio.gather(*(_generate(client, pd_service[0].public_url) for _ in range(12)))
            assert all(response.status_code == 200 for response in responses)
            ids = [req["params"]["group_request_id"] for req in peers.prefill_reqs]
            assert len(ids) == len(set(ids)) == 12
            assert {req["master_id"] for req in peers.prefill_reqs} == {w["node_id"] for w in pd_service[0].workers}
            by_id = {req["params"]["group_request_id"]: req for req in peers.prefill_reqs}
            for req in peers.decode_reqs:
                prefill = by_id[req["params"]["group_request_id"]]
                assert req["master_id"] == prefill["master_id"]
                assert req["prompt_ids"] == prefill["prompt_ids"]

    _run(pd_service, scenario)
