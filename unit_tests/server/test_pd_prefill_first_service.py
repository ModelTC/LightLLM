"""Service tests for PD 分离的 Prefill 先推理模式.

Starts one real PD master with two workers. Prefill and decode are protocol
peers on the websocket, so the scenarios do not load a model.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import pickle

import httpx
import pytest
import websockets
from prometheus_client.parser import text_string_to_metric_families
from websockets.exceptions import ConnectionClosed

from lightllm.server.core.objs import FinishStatus
from lightllm.server.pd_io_struct import ObjType, PDDecodeNodeInfo, PDUpKVStatus, PdPfPrefillKvReady
from unit_tests.server.pd_master_test_utils import running_pd_master


@dataclass
class Segment:
    prompt_ids: list = None
    prefill_text: str = "hello"
    prefill_finish: int = FinishStatus.NO_FINISH
    prefill_cache_len: int = 0
    decode_steps: tuple = None
    ready_kv_len: int = 0
    token_before_kv_ready: bool = True
    hold_prefill: bool = False
    hold_upkv: bool = False
    busy: bool = False
    prefill_error: str = ""
    decode_error: str = ""
    close_prefill: bool = False

    def steps(self):
        if self.decode_steps is None:
            return (
                ("hello", FinishStatus.NO_FINISH, 1),
                (" world", FinishStatus.FINISHED_STOP, 2),
            )
        return self.decode_steps


def _capacity_segments():
    return [
        Segment(
            prompt_ids=[4],
            prefill_text="A",
            decode_steps=(
                ("A", FinishStatus.NO_FINISH, 1),
                ("B", FinishStatus.FINISHED_PD_DECODE_CAPACITY, 2),
            ),
        ),
        Segment(
            prompt_ids=[4, 5],
            prefill_text="C",
            decode_steps=(
                ("C", FinishStatus.NO_FINISH, 1),
                ("D", FinishStatus.FINISHED_STOP, 2),
            ),
        ),
    ]


def _two_nodes():
    return (
        ("prefill", "prefill-a", "prefill-a.internal:18001", 501),
        ("prefill", "prefill-b", "prefill-b.internal:18011", 511),
        ("decode", "decode-a", "decode-a.internal:18002", 502),
        ("decode", "decode-b", "decode-b.internal:18012", 512),
    )


class PrefillFirstPeers:
    """Register protocol peers on every selected PD master worker."""

    def __init__(self, master, segments=None, nodes=None):
        self.master = master
        self.segments = segments or [Segment()]
        self.nodes = nodes
        self.prefill_reqs = []
        self.decode_reqs = []
        self.kv_ready = []
        self.aborts = []
        self.message_types = []
        self.upkv_sent = []
        self.prefill_sends = []
        self.peer_errors = []
        self.release_prefill = asyncio.Event()
        self.release_upkv = asyncio.Event()
        self._prefill_ordinal = {}
        self._group_segments = {}
        self._tasks = []
        self._send_locks = {}
        self._stack = None
        self._closing = False
        if not any(segment.hold_prefill for segment in self.segments):
            self.release_prefill.set()
        if not any(segment.hold_upkv for segment in self.segments):
            self.release_upkv.set()

    async def connect(self, workers=None, nodes=None, wait_ready=True):
        workers = self.master.workers if workers is None else workers
        nodes = self.nodes if nodes is None else nodes
        if nodes is None:
            nodes = (
                ("prefill", "prefill", "prefill.internal:18001", 501),
                ("decode", "decode", "decode.internal:18002", 502),
            )
        if self._stack is None:
            from contextlib import AsyncExitStack

            self._stack = AsyncExitStack()
            await self._stack.__aenter__()
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            for worker in workers:
                response = await client.get("http://" + worker["host_ip_port"] + "/get_server_info")
                response.raise_for_status()
                start_args = response.json()
                for mode, node_name, address, node_id in nodes:
                    websocket = await self._stack.enter_async_context(
                        websockets.connect(
                            "ws://" + worker["host_ip_port"] + "/pd_register",
                            proxy=None,
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "node_id": node_id,
                                "client_ip_port": address,
                                "mode": mode,
                                "start_args": dict(start_args, run_mode=mode),
                                "connection_id": f"{node_name}-{worker['node_id']}",
                            }
                        )
                    )
                    self._track(
                        self._run(
                            mode=mode,
                            master_id=worker["node_id"],
                            node_name=node_name,
                            websocket=websocket,
                        )
                    )
                    self._track(self._heartbeat(websocket=websocket))
        if wait_ready:
            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                for worker in workers:
                    await _wait_status(
                        client=client,
                        url="http://" + worker["host_ip_port"] + "/readiness",
                        expected=200,
                        timeout=10,
                    )
        return self

    async def close(self):
        self._closing = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    def _track(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.append(task)

        def _done(done_task):
            if done_task.cancelled():
                return
            error = done_task.exception()
            if error is not None and not isinstance(error, (ConnectionClosed, asyncio.CancelledError)):
                self.peer_errors.append(error)

        task.add_done_callback(_done)
        return task

    async def _send(self, websocket, message):
        lock = self._send_locks.setdefault(id(websocket), asyncio.Lock())
        async with lock:
            await websocket.send(pickle.dumps(message))

    async def _heartbeat(self, websocket):
        while True:
            await asyncio.sleep(5)
            await self._send(websocket=websocket, message=(ObjType.HEARTBEAT,))

    async def _run(self, mode, master_id, node_name, websocket):
        async for data in websocket:
            message = pickle.loads(data)
            self.message_types.append((mode, master_id, message[0]))
            if message[0] == ObjType.ABORT:
                self.aborts.append((mode, master_id, message[1]))
                continue
            if mode == "prefill" and message[0] == ObjType.PD_PF_PREFILL_REQ:
                self._track(
                    self._reply_prefill(
                        websocket=websocket,
                        master_id=master_id,
                        node_name=node_name,
                        message=message,
                    )
                )
                continue
            if mode == "prefill" and message[0] == ObjType.PD_PF_DECODE_KV_READY:
                _, group_request_id, info = message
                self.kv_ready.append((master_id, group_request_id, info))
                continue
            if mode == "decode" and message[0] == ObjType.REQ:
                self._track(
                    self._reply_decode(
                        websocket=websocket,
                        master_id=master_id,
                        node_name=node_name,
                        message=message,
                    )
                )

    def _segment_for_prefill(self, master_id, group_request_id):
        # 续跑会换成新的 group id。按这个 master 上的 prefill 序号取段，decode 再按 group id 找回同一段。
        ordinal = self._prefill_ordinal.get(master_id, 0)
        self._prefill_ordinal[master_id] = ordinal + 1
        segment = self.segments[min(ordinal, len(self.segments) - 1)]
        self._group_segments[group_request_id] = segment
        return segment

    async def _reply_prefill(self, websocket, master_id, node_name, message):
        _, (prompt, params, _) = message
        view = _params_view(params)
        segment = self._segment_for_prefill(master_id=master_id, group_request_id=view["group_request_id"])
        prompt_ids = segment.prompt_ids if segment.prompt_ids is not None else [len(prompt), 2, 3]
        self.prefill_reqs.append(
            {
                "master_id": master_id,
                "node_name": node_name,
                "prompt": prompt,
                "prompt_ids": list(prompt_ids),
                "params": view,
            }
        )
        if segment.close_prefill:
            await websocket.close()
            return
        if segment.hold_prefill:
            await self.release_prefill.wait()
        group_request_id = view["group_request_id"]
        if segment.busy:
            await self._send(
                websocket=websocket,
                message=(ObjType.PD_UPLOAD_SERVER_BUSY, group_request_id, "test capacity exhausted"),
            )
            return
        if segment.prefill_error:
            await self._send(
                websocket=websocket,
                message=(ObjType.PD_UPLOAD_GENERATE_ERROR, group_request_id, segment.prefill_error),
            )
            return
        ready = PdPfPrefillKvReady(group_request_id=group_request_id, prompt_ids=list(prompt_ids))
        token = _token_message(
            mode="prefill",
            master_id=master_id,
            request_id=group_request_id,
            text=segment.prefill_text,
            finish=segment.prefill_finish,
            count=1,
            cache_len=segment.prefill_cache_len,
        )
        kv_ready = (ObjType.PD_PF_PREFILL_KV_READY, ready)
        if segment.token_before_kv_ready:
            self.prefill_sends.append(ObjType.TOKEN_PACKS)
            await self._send(websocket=websocket, message=token)
            self.prefill_sends.append(ObjType.PD_PF_PREFILL_KV_READY)
            await self._send(websocket=websocket, message=kv_ready)
        else:
            self.prefill_sends.append(ObjType.PD_PF_PREFILL_KV_READY)
            await self._send(websocket=websocket, message=kv_ready)
            self.prefill_sends.append(ObjType.TOKEN_PACKS)
            await self._send(websocket=websocket, message=token)

    async def _reply_decode(self, websocket, master_id, node_name, message):
        _, (prompt_ids, params, _) = message
        view = _params_view(params)
        segment = self._group_segments[view["group_request_id"]]
        self.decode_reqs.append(
            {
                "master_id": master_id,
                "node_name": node_name,
                "prompt_ids": list(prompt_ids),
                "params": view,
            }
        )
        if segment.hold_upkv:
            await self.release_upkv.wait()
        group_request_id = view["group_request_id"]
        info = PDDecodeNodeInfo(
            decode_node_id=702,
            pd_master_node_id=master_id,
            agent_name=f"decode-{master_id}-{group_request_id}",
            agent_metadata=b"meta",
            num_pages=1,
            page_reg_desc=b"pages",
            request_id=group_request_id,
            ready_kv_len=segment.ready_kv_len,
        )
        self.upkv_sent.append(group_request_id)
        await self._send(
            websocket=websocket,
            message=(
                ObjType.PD_UPLOAD_KV_STATUS,
                PDUpKVStatus(
                    group_request_id=group_request_id,
                    pd_master_node_id=master_id,
                    pd_kv_trans_params=pickle.dumps(info),
                ),
            ),
        )
        if segment.decode_error:
            await _wait_until(
                lambda: any(item[1] == group_request_id for item in self.kv_ready),
                timeout=5,
            )
            await self._send(
                websocket=websocket,
                message=(ObjType.PD_UPLOAD_GENERATE_ERROR, group_request_id, segment.decode_error),
            )
            return
        for text, finish, count in segment.steps():
            await self._send(
                websocket=websocket,
                message=_token_message(
                    mode="decode",
                    master_id=master_id,
                    request_id=group_request_id,
                    text=text,
                    finish=finish,
                    count=count,
                    cache_len=0,
                ),
            )


def _params_view(params):
    return {
        "max_new_tokens": int(params.max_new_tokens),
        "pd_prefill_first": bool(params.pd_prefill_first),
        "pd_master_node_id": int(params.pd_master_node_id.get()),
        "group_request_id": int(params.group_request_id),
        "infer_high_priority": int(params.infer_high_priority),
        "temperature": float(params.temperature),
        "do_sample": bool(params.do_sample),
    }


def _token_message(mode, master_id, request_id, text, finish, count, cache_len):
    metadata = {
        "id": 2,
        "logprob": -0.1,
        "logprobs": None,
        "prompt_tokens": 3,
        "count_output_tokens": count,
        "node_mode": mode,
        "prompt_cache_len": cache_len,
    }
    return (
        ObjType.TOKEN_PACKS,
        [(request_id, text, metadata, FinishStatus(finish))],
        {"connection_key": f"ignored-{master_id}", "total_token_usage_rate": 0.1},
    )


def _run(master, scenario):
    try:
        asyncio.run(scenario(master))
    except BaseException:
        print(master.log_path.read_text(errors="replace")[-12000:])
        raise


async def _wait_until(predicate, timeout=5):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting for a protocol event"
        await asyncio.sleep(0.02)


async def _wait_status(client, url, expected, timeout=20):
    deadline = asyncio.get_running_loop().time() + timeout
    last = None
    while True:
        last = await client.get(url)
        if last.status_code == expected:
            return last
        assert asyncio.get_running_loop().time() < deadline, last.text
        await asyncio.sleep(0.1)


async def _generate(client, url, prompt="hello", max_new_tokens=4, extra=None):
    parameters = {"max_new_tokens": max_new_tokens}
    if extra:
        parameters.update(extra)
    return await client.post(url + "/generate", json={"inputs": prompt, "parameters": parameters})


def _assert_ok(response, text):
    assert response.status_code == 200, response.text
    assert response.json()["generated_text"] == text


@asynccontextmanager
async def _open(master, segments=None, nodes=None, workers=None, wait_ready=True):
    peers = PrefillFirstPeers(master=master, segments=segments, nodes=nodes)
    try:
        await peers.connect(workers=workers, nodes=nodes, wait_ready=wait_ready)
        async with httpx.AsyncClient(trust_env=False, timeout=20) as client:
            yield peers, client
    finally:
        await peers.close()


@pytest.fixture(scope="module")
def prefill_first_service(tmp_path_factory):
    runtime_dir = tmp_path_factory.mktemp("prefill-first-masters")
    with running_pd_master(
        runtime_dir,
        extra_args=[
            "--httpserver_workers",
            "2",
            "--pd_prefill_first",
            "--select_p_d_node_strategy",
            "round_robin",
        ],
    ) as master:
        assert len(master.workers) == 2
        yield master


def test_01_two_masters_stay_unready_until_each_has_prefill_and_decode(prefill_first_service):
    async def scenario(master):
        peers = PrefillFirstPeers(master=master)
        try:
            await peers.connect(workers=master.workers[:1], wait_ready=True)
            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                deadline = asyncio.get_running_loop().time() + 8
                while asyncio.get_running_loop().time() < deadline:
                    response = await client.get(master.public_url + "/readiness")
                    assert response.status_code == 503, response.text
                    await asyncio.sleep(0.5)
                await peers.connect(workers=master.workers[1:], wait_ready=True)
                ready = await _wait_status(
                    client=client,
                    url=master.public_url + "/readiness",
                    expected=200,
                    timeout=20,
                )
                assert ready.json()["status"] == "ok"
        finally:
            await peers.close()

    _run(prefill_first_service, scenario)


def test_02_prefill_receives_pf_req_not_the_old_handshake(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert [item[2] for item in peers.message_types if item[0] == "prefill" and item[2] == ObjType.REQ] == []
            assert ObjType.PD_PF_PREFILL_REQ in [item[2] for item in peers.message_types]
            assert ObjType.PD_REQ_DECODE_NODE_INFO not in [item[2] for item in peers.message_types]

    _run(prefill_first_service, scenario)


def test_03_prefill_sampling_marks_prefill_first(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert peers.prefill_reqs[0]["params"]["pd_prefill_first"] is True

    _run(prefill_first_service, scenario)


def test_04_prefill_max_new_tokens_is_temporarily_one(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url, max_new_tokens=4)
            _assert_ok(response, ["hello world"])
            assert peers.prefill_reqs[0]["params"]["max_new_tokens"] == 1

    _run(prefill_first_service, scenario)


def test_05_decode_restores_the_client_max_new_tokens(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url, max_new_tokens=4)
            _assert_ok(response, ["hello world"])
            assert peers.decode_reqs[0]["params"]["max_new_tokens"] == 4

    _run(prefill_first_service, scenario)


def test_06_both_sides_carry_the_serving_master_node_id(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            prefill = peers.prefill_reqs[0]
            decode = peers.decode_reqs[0]
            assert prefill["params"]["pd_master_node_id"] == prefill["master_id"]
            assert decode["params"]["pd_master_node_id"] == decode["master_id"]
            assert prefill["master_id"] == decode["master_id"]

    _run(prefill_first_service, scenario)


def test_07_prefill_receives_the_original_prompt(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url, prompt="hello world")
            _assert_ok(response, ["hello world"])
            assert peers.prefill_reqs[0]["prompt"] == "hello world"

    _run(prefill_first_service, scenario)


def test_08_decode_receives_prompt_ids_from_prefill_kv_ready(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(prompt_ids=[9, 8, 7])]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert peers.decode_reqs[0]["prompt_ids"] == [9, 8, 7]

    _run(prefill_first_service, scenario)


def test_09_stream_emits_the_prefill_token_before_decode_upkv(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(hold_upkv=True)]) as (peers, client):
            async with client.stream(
                "POST",
                master.public_url + "/generate_stream",
                json={"inputs": "hello", "parameters": {"max_new_tokens": 4}},
            ) as response:
                assert response.status_code == 200
                events = []
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    events.append(json.loads(line[5:]))
                    if len(events) == 1:
                        assert peers.upkv_sent == []
                        peers.release_upkv.set()
            assert [event["token"]["text"] for event in events] == ["hello", " world"]

    _run(prefill_first_service, scenario)


def test_10_decode_repeat_of_the_first_token_is_omitted(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            async with client.stream(
                "POST",
                master.public_url + "/generate_stream",
                json={"inputs": "hello", "parameters": {"max_new_tokens": 4}},
            ) as response:
                assert response.status_code == 200
                events = [json.loads(line[5:]) async for line in response.aiter_lines() if line.startswith("data:")]
            assert [event["token"]["text"] for event in events] == ["hello", " world"]
            assert peers.decode_reqs

    _run(prefill_first_service, scenario)


def test_11_generate_joins_prefill_and_later_decode_tokens(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert peers.prefill_reqs and peers.decode_reqs

    _run(prefill_first_service, scenario)


def test_12_stream_finish_flags_follow_the_last_token(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            async with client.stream(
                "POST",
                master.public_url + "/generate_stream",
                json={"inputs": "hello", "parameters": {"max_new_tokens": 4}},
            ) as response:
                events = [json.loads(line[5:]) async for line in response.aiter_lines() if line.startswith("data:")]
            assert [event["finished"] for event in events] == [False, True]
            assert peers.kv_ready

    _run(prefill_first_service, scenario)


def test_13_stop_finish_reason_reaches_the_client(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert response.json()["finish_reason"] == "stop"

    _run(prefill_first_service, scenario)


def test_14_one_token_request_finishes_on_prefill_and_skips_decode(prefill_first_service):
    async def scenario(master):
        async with _open(
            master,
            segments=[Segment(prefill_text="hello", prefill_finish=FinishStatus.FINISHED_LENGTH)],
        ) as (peers, client):
            response = await _generate(client=client, url=master.public_url, max_new_tokens=1)
            _assert_ok(response, ["hello"])
            assert response.json()["finish_reason"] == "length"
            assert peers.decode_reqs == []
            await _wait_until(lambda: any(item[0] == "prefill" for item in peers.aborts))

    _run(prefill_first_service, scenario)


def test_15_prefill_length_finish_is_rewritten_when_more_tokens_remain(prefill_first_service):
    async def scenario(master):
        async with _open(
            master,
            segments=[Segment(prefill_text="hello", prefill_finish=FinishStatus.FINISHED_LENGTH)],
        ) as (peers, client):
            response = await _generate(client=client, url=master.public_url, max_new_tokens=4)
            _assert_ok(response, ["hello world"])
            assert len(peers.decode_reqs) == 1
            assert response.json()["finish_reason"] == "stop"

    _run(prefill_first_service, scenario)


def test_16_decode_kv_ready_uses_the_prefill_first_message(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert [item[2] for item in peers.message_types].count(ObjType.PD_PF_DECODE_KV_READY) == 1
            assert ObjType.PD_REQ_DECODE_NODE_INFO not in [item[2] for item in peers.message_types]

    _run(prefill_first_service, scenario)


def test_17_ready_kv_len_is_forwarded_unchanged(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(ready_kv_len=4, prompt_ids=[1, 2, 3, 4, 5, 6])]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert peers.kv_ready[0][2].ready_kv_len == 4

    _run(prefill_first_service, scenario)


def test_18_full_match_ready_kv_len_still_leaves_the_reserved_token(prefill_first_service):
    async def scenario(master):
        prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        async with _open(
            master,
            segments=[Segment(ready_kv_len=len(prompt_ids) - 1, prompt_ids=prompt_ids)],
        ) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            info = peers.kv_ready[0][2]
            assert info.ready_kv_len == 7
            assert peers.decode_reqs[0]["prompt_ids"] == prompt_ids

    _run(prefill_first_service, scenario)


def test_19_zero_ready_kv_len_is_forwarded(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(ready_kv_len=0)]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert peers.kv_ready[0][2].ready_kv_len == 0

    _run(prefill_first_service, scenario)


def test_20_requests_land_on_both_masters(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):

            async def one():
                response = await _generate(client=client, url=master.public_url)
                _assert_ok(response, ["hello world"])

            await asyncio.gather(*(one() for _ in range(8)))
            assert {item["master_id"] for item in peers.prefill_reqs} == {
                worker["node_id"] for worker in master.workers
            }

    _run(prefill_first_service, scenario)


def test_21_one_request_stays_on_a_single_master(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert peers.prefill_reqs[0]["master_id"] == peers.decode_reqs[0]["master_id"]
            assert peers.kv_ready[0][0] == peers.prefill_reqs[0]["master_id"]

    _run(prefill_first_service, scenario)


def test_22_request_ids_from_two_masters_do_not_collide(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):

            async def one():
                response = await _generate(client=client, url=master.public_url)
                _assert_ok(response, ["hello world"])

            await asyncio.gather(*(one() for _ in range(8)))
            ids = [item["params"]["group_request_id"] for item in peers.prefill_reqs]
            assert len(ids) == len(set(ids))
            assert {item["master_id"] for item in peers.prefill_reqs} == {
                worker["node_id"] for worker in master.workers
            }

    _run(prefill_first_service, scenario)


def test_23_client_cancel_after_the_first_token_aborts_both_sides(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(hold_upkv=True)]) as (peers, client):
            async with client.stream(
                "POST",
                master.public_url + "/generate_stream",
                json={"inputs": "hello", "parameters": {"max_new_tokens": 4}},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        assert json.loads(line[5:])["token"]["text"] == "hello"
                        break
            await _wait_until(lambda: peers.decode_reqs)
            master_id = peers.prefill_reqs[0]["master_id"]
            request_id = peers.prefill_reqs[0]["params"]["group_request_id"]
            await _wait_until(
                lambda: ("prefill", master_id, request_id) in peers.aborts
                and ("decode", master_id, request_id) in peers.aborts
            )

    _run(prefill_first_service, scenario)


def test_24_cancel_before_prefill_kv_ready_does_not_open_decode(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(hold_prefill=True)]) as (peers, client):
            pending = asyncio.create_task(_generate(client=client, url=master.public_url))
            await _wait_until(lambda: bool(peers.prefill_reqs))
            master_id = peers.prefill_reqs[0]["master_id"]
            request_id = peers.prefill_reqs[0]["params"]["group_request_id"]
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await _wait_until(lambda: ("prefill", master_id, request_id) in peers.aborts)
            assert peers.decode_reqs == []

    _run(prefill_first_service, scenario)


def test_25_prefill_busy_returns_429_without_decode(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(busy=True)]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            assert response.status_code == 429, response.text
            assert response.json()["error"]["type"] == "RateLimitError"
            assert peers.decode_reqs == []

    _run(prefill_first_service, scenario)


def test_26_prefill_generate_error_returns_417(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(prefill_error="prefill exploded")]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            assert response.status_code == 417, response.text
            assert "prefill exploded" in response.text
            assert peers.decode_reqs == []

    _run(prefill_first_service, scenario)


def test_27_decode_generate_error_fails_the_http_request(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(decode_error="decode exploded")]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            assert response.status_code == 417, response.text
            assert "decode exploded" in response.text
            assert peers.kv_ready

    _run(prefill_first_service, scenario)


def test_28_decode_capacity_continues_on_the_same_prefill(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=_capacity_segments()) as (peers, client):
            response = await _generate(client=client, url=master.public_url, prompt="hello", max_new_tokens=3)
            _assert_ok(response, ["ACD"])
            assert [item["prompt"] for item in peers.prefill_reqs] == ["hello", "helloA"]
            assert peers.prefill_reqs[0]["master_id"] == peers.prefill_reqs[1]["master_id"]
            assert peers.prefill_reqs[0]["node_name"] == peers.prefill_reqs[1]["node_name"]

    _run(prefill_first_service, scenario)


def test_29_capacity_marker_text_is_absent_from_the_client_output(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=_capacity_segments()) as (peers, client):
            response = await _generate(client=client, url=master.public_url, prompt="hello", max_new_tokens=3)
            _assert_ok(response, ["ACD"])
            assert "B" not in response.json()["generated_text"][0]
            assert len(peers.decode_reqs) == 2

    _run(prefill_first_service, scenario)


def test_30_continuation_segment_is_marked_infer_high_priority(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=_capacity_segments()) as (peers, client):
            response = await _generate(client=client, url=master.public_url, prompt="hello", max_new_tokens=3)
            _assert_ok(response, ["ACD"])
            assert peers.prefill_reqs[0]["params"]["infer_high_priority"] == 0
            assert peers.prefill_reqs[1]["params"]["infer_high_priority"] == -1
            assert peers.decode_reqs[1]["params"]["max_new_tokens"] == 2

    _run(prefill_first_service, scenario)


def test_31_kv_ready_arriving_before_the_prefill_token_still_completes(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(token_before_kv_ready=False)]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert peers.prefill_sends[:2] == [ObjType.PD_PF_PREFILL_KV_READY, ObjType.TOKEN_PACKS]

    _run(prefill_first_service, scenario)


def test_32_concurrent_requests_keep_their_own_prompt_ids(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):

            async def one(prompt):
                response = await _generate(client=client, url=master.public_url, prompt=prompt)
                _assert_ok(response, ["hello world"])

            await asyncio.gather(one("alpha"), one("beta"))
            by_id = {item["params"]["group_request_id"]: item for item in peers.prefill_reqs}
            assert {item["prompt"] for item in peers.prefill_reqs} == {"alpha", "beta"}
            for decode in peers.decode_reqs:
                prefill = by_id[decode["params"]["group_request_id"]]
                assert decode["prompt_ids"] == prefill["prompt_ids"]
                assert decode["master_id"] == prefill["master_id"]

    _run(prefill_first_service, scenario)


def test_33_two_choices_open_two_independent_prefill_requests(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(
                client=client,
                url=master.public_url,
                extra={"n": 2, "best_of": 2},
            )
            _assert_ok(response, ["hello world", "hello world"])
            assert len(peers.prefill_reqs) == 2
            assert len({item["params"]["group_request_id"] for item in peers.prefill_reqs}) == 2

    _run(prefill_first_service, scenario)


def test_34_sampling_temperature_reaches_prefill(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(
                client=client,
                url=master.public_url,
                extra={"temperature": 0.3, "do_sample": True},
            )
            _assert_ok(response, ["hello world"])
            assert peers.prefill_reqs[0]["params"]["do_sample"] is True
            assert abs(peers.prefill_reqs[0]["params"]["temperature"] - 0.3) < 1e-4

    _run(prefill_first_service, scenario)


def test_35_old_decode_node_info_message_is_not_sent(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            assert all(item[2] != ObjType.PD_REQ_DECODE_NODE_INFO for item in peers.message_types)
            assert peers.kv_ready[0][2].agent_name.startswith("decode-")

    _run(prefill_first_service, scenario)


def test_36_a_second_request_reuses_the_live_connections(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            first = await _generate(client=client, url=master.public_url, prompt="hello")
            second = await _generate(client=client, url=master.public_url, prompt="world")
            _assert_ok(first, ["hello world"])
            _assert_ok(second, ["hello world"])
            assert [item["prompt"] for item in peers.prefill_reqs] == ["hello", "world"]
            assert not peers.peer_errors

    _run(prefill_first_service, scenario)


def test_37_prefill_disconnect_fails_the_in_flight_request(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(close_prefill=True)]) as (peers, client):
            response = await _generate(client=client, url=master.public_url)
            assert response.status_code == 417, response.text
            assert peers.decode_reqs == []

    _run(prefill_first_service, scenario)


def test_38_round_robin_second_select_pins_prefill_and_advances_decode(prefill_first_service):
    async def scenario(master):
        async with _open(master, nodes=_two_nodes()) as (peers, client):
            for _ in range(3):
                response = await _generate(client=client, url=master.public_url)
                _assert_ok(response, ["hello world"])
            by_master = {}
            for item in peers.prefill_reqs:
                by_master.setdefault(item["master_id"], {"prefill": [], "decode": []})
                by_master[item["master_id"]]["prefill"].append(item["node_name"])
            for item in peers.decode_reqs:
                by_master[item["master_id"]]["decode"].append(item["node_name"])
            assert by_master
            for names in by_master.values():
                assert len(set(names["prefill"])) == 1
                assert len(set(names["decode"])) == 1
                assert names["prefill"][0].rsplit("-", 1)[-1] != names["decode"][0].rsplit("-", 1)[-1]

    _run(prefill_first_service, scenario)


def test_39_prompt_cache_len_from_prefill_is_kept_for_later_tokens(prefill_first_service):
    async def scenario(master):
        async with _open(master, segments=[Segment(prefill_cache_len=7)]) as (peers, client):
            response = await _generate(client=client, url=master.public_url, extra={"return_details": True})
            _assert_ok(response, ["hello world"])
            tokens = response.json()["tokens"]
            assert [token["prompt_cache_len"] for token in tokens] == [7, 7]
            assert peers.decode_reqs

    _run(prefill_first_service, scenario)


def test_40_request_counter_increases_once_for_one_request(prefill_first_service):
    async def scenario(master):
        async with _open(master) as (peers, client):
            before = [await _request_count(client, worker) for worker in master.workers]
            response = await _generate(client=client, url=master.public_url)
            _assert_ok(response, ["hello world"])
            after = [await _request_count(client, worker) for worker in master.workers]
            assert [end - start for start, end in zip(before, after)] == [1, 1]
            assert peers.prefill_reqs

    _run(prefill_first_service, scenario)


async def _request_count(client, worker):
    response = await client.get("http://" + worker["host_ip_port"] + "/metrics")
    assert response.status_code == 200, response.text
    values = [
        sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == "lightllm_request_count_total"
    ]
    return values[0] if values else 0.0
