import asyncio
import random
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import numpy as np

from lightllm.server.core.objs import StopStrMatchMode, FinishStatus, SamplingParams, Req
from lightllm.server.core.objs.out_token_circlequeue import CircularQueue
from lightllm.server.detokenization.decode_req import DecodeReq
from lightllm.server.detokenization.manager import DeTokenizationManager
from lightllm.server.httpserver.stop_seq_match import (
    StopStrMatchBuffer,
    StopTokenIdsMatchBuffer,
    match_stop_token_ids_stream,
    match_stop_str_stream,
    wrap_stop_match_stream,
)
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.req_id_generator import convert_sub_id_to_group_id


class _ReasoningTokenizer:
    def encode(self, text, **kwargs):
        return {"<think>": [1000], "</think>": [1001]}.get(text, [ord(c) for c in text])


def _params(stop="END", include=False, force=0):
    params = SamplingParams()
    params.init(
        _ReasoningTokenizer(),
        stop_sequences=stop,
        _initial_reasoning_state=force,
        include_stop_str_in_output=include,
        stop_str_match_mode=StopStrMatchMode.ALL_TEXT,
    )
    return params


def _stop_buffer(stop_in_reasoning=False, stop="END", force=1, parser="qwen3", include=False):
    params = _params(stop, include, force)
    if not stop_in_reasoning:
        params.stop_str_match_mode.set_mode(StopStrMatchMode.CONTENT_ONLY)
    manager = _http_manager()
    manager.args.reasoning_parser = parser
    return StopStrMatchBuffer(manager, params, [1, 2])


def _output(token_id, text, payload, finish=FinishStatus.NO_FINISH):
    return 80, text, {"id": token_id, "payload": payload}, FinishStatus(finish)


def _handle_output(buffer, *output):
    assert asyncio.run(buffer.handle(*output)) is None


def _drain_ready_tokens(buffer):
    tokens = []
    while buffer.ready_token_list:
        req_id, text, metadata, _finish = buffer.ready_token_list.popleft()
        assert req_id == 80
        tokens.append((text, metadata["payload"]))
    return tokens


def _req(tokens, finish=FinishStatus.NO_FINISH, params=None, request_id=80):
    return SimpleNamespace(
        request_id=request_id,
        group_req_id=80,
        input_len=2,
        sample_params=params or _params(),
        shm_prompt_ids=SimpleNamespace(arr=np.array([1, 2, *tokens])),
        shm_logprobs=SimpleNamespace(arr={"logprob": np.array([0, 0, *([-0.5] * len(tokens))])}),
        finish_token_index=1 + len(tokens) if finish != FinishStatus.NO_FINISH else -1,
        finish_status=FinishStatus(finish),
        is_aborted=False,
        candetoken_out_len=len(tokens),
        out_tokens_queue=CircularQueue(),
        can_released_mark=False,
        ref_count=1,
        cumlogprob=0,
        prompt_cache_len=0,
        cpu_prompt_cache_len=0,
        disk_prompt_cache_len=0,
        mtp_accepted_token_num=0,
        mtp_verify_token_num=0,
        mtp_verify_step_num=0,
        get_output_logprobs_metadata=lambda *args: {},
        merge_final_token_metadata=AsyncMock(),
    )


def _http_manager():
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.tokenizer = _ReasoningTokenizer()
    manager.args = SimpleNamespace(
        use_reward_model=False, enable_rl=False, enable_return_routed_experts=False, reasoning_parser=None
    )
    manager.pd_mode = SimpleNamespace(is_normal=lambda: True)
    manager.is_multinode_tp_slave = False
    manager.req_id_to_out_inf = {}
    return manager


async def _read_http_tokens(manager, req):
    """运行一轮实际 HTTP 读取循环，随后停止循环并取出打包结果。"""
    status = SimpleNamespace(
        group_req_objs=SimpleNamespace(shm_req_objs=[req]),
        out_token_info_list=[],
        lock=asyncio.Lock(),
        event=asyncio.Event(),
    )
    previous_requests = manager.req_id_to_out_inf
    manager.req_id_to_out_inf = {convert_sub_id_to_group_id(req.request_id): status}
    manager.recycle_resource_loop = AsyncMock()
    manager.pd_mode.is_P_or_D = lambda: False
    manager.zmq_recv_socket = SimpleNamespace(recv_pyobj=AsyncMock(side_effect=[None, StopAsyncIteration()]))
    try:
        with pytest.raises(StopAsyncIteration):
            await manager.handle_loop()
    finally:
        manager.req_id_to_out_inf = previous_requests
    return status.out_token_info_list


def _match_http_stream(manager, generator, params, prompt_ids, reqs):
    for req in reqs:
        manager.req_id_to_out_inf[convert_sub_id_to_group_id(req.request_id)] = SimpleNamespace(
            group_req_objs=SimpleNamespace(shm_req_objs=reqs)
        )
    return match_stop_str_stream(manager, generator, params, prompt_ids)


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize(
    "stops,chunks,prefix,matched_stop",
    [
        (["END"], ["helloENDextra"], "hello", "END"),
        (["END"], ["helloE"] + [""] * 30 + ["NDextra"], "hello", "END"),
        (["LATER", "END"], ["helloENDLATER"], "hello", "END"),
        (["\n", "\n\n"], ["hello\n\nextra"], "hello", "\n\n"),
        (["结束"], ["你好结", "束后缀"], "你好", "结束"),
    ],
)
def test_stop_buffer_trims_text_and_preserves_token_payloads(include, stops, chunks, prefix, matched_stop):
    buffer = StopStrMatchBuffer(_http_manager(), _params(stops, include), [])
    payloads = [object() for _ in chunks]
    results = []
    for index, (text, payload) in enumerate(zip(chunks, payloads)):
        _handle_output(buffer, *_output(index, text, payload))
        ready_tokens = _drain_ready_tokens(buffer)
        results.extend(ready_tokens)
        assert (buffer.state is buffer.State.STOPPED) is (index == len(chunks) - 1)
    assert "".join(text for text, _ in results) == prefix + (matched_stop if include else "")
    assert [payload for _, payload in results] == payloads
    assert buffer.state is buffer.State.STOPPED


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("backend_finished", [False, True])
def test_stop_buffer_accepts_generator_output_and_preserves_metadata(include, backend_finished):
    manager = _http_manager()
    buffer = StopStrMatchBuffer(manager, _params(include=include), [])

    async def stop_request(req_id):
        assert req_id == 81
        assert buffer.state is buffer.State.STOPPED
        assert buffer.ready_token_list[-1][3].is_stopped()

    manager.stop_request = AsyncMock(side_effect=stop_request)
    metadata = [{"id": 10, "logprob": -0.1}, {"id": 11, "logprob": -0.2}]
    raw_finishes = [
        FinishStatus(),
        FinishStatus(FinishStatus.FINISHED_LENGTH if backend_finished else FinishStatus.NO_FINISH),
    ]
    _handle_output(buffer, 81, "helloE", metadata[0], raw_finishes[0])
    manager.stop_request.assert_not_awaited()
    _handle_output(buffer, 81, "NDextra", metadata[1], raw_finishes[1])
    _handle_output(buffer, 81, "lateEND", {"id": 12}, FinishStatus())
    if backend_finished:
        manager.stop_request.assert_not_awaited()
    else:
        manager.stop_request.assert_awaited_once_with(81)
    first, last = buffer.ready_token_list.popleft(), buffer.ready_token_list.popleft()
    assert first[:2] == (81, "helloE" if include else "hello")
    assert last[:2] == (81, "ND" if include else "")
    assert first[2] is metadata[0] and last[2] is metadata[1]
    assert first[3] is not raw_finishes[0] and last[3] is not raw_finishes[1]
    assert not first[3].is_finished() and last[3].is_stopped()
    assert raw_finishes[0].status == FinishStatus.NO_FINISH
    assert raw_finishes[1].status == (FinishStatus.FINISHED_LENGTH if backend_finished else FinishStatus.NO_FINISH)
    assert not buffer.ready_token_list


@pytest.mark.parametrize("include", [False, True])
def test_only_possible_stop_prefix_delays_whole_token(include):
    buffer = StopStrMatchBuffer(_http_manager(), _params(include=include), [])
    assert buffer.state is buffer.State.SCANNING
    _handle_output(buffer, *_output(1, "hello", 1))
    ready_tokens = _drain_ready_tokens(buffer)
    assert ready_tokens == [("hello", 1)]
    assert buffer.state is buffer.State.SCANNING
    _handle_output(buffer, *_output(2, "worldE", 2))
    ready_tokens = _drain_ready_tokens(buffer)
    assert not ready_tokens
    assert buffer.state is buffer.State.WAITING
    _handle_output(buffer, *_output(3, "X", 3))
    ready_tokens = _drain_ready_tokens(buffer)
    assert ready_tokens == [("worldE", 2), ("X", 3)]
    assert buffer.state is buffer.State.SCANNING


@pytest.mark.parametrize("include", [False, True])
def test_releasing_one_token_preserves_next_token_before_stop(include):
    buffer = StopStrMatchBuffer(_http_manager(), _params(include=include), [])
    _handle_output(buffer, *_output(1, "aE", "first"))
    assert not buffer.ready_token_list
    assert buffer.state is buffer.State.WAITING
    _handle_output(buffer, *_output(2, "bE", "second"))
    released_tokens = _drain_ready_tokens(buffer)
    assert released_tokens == [("aE", "first")]
    assert buffer.state is buffer.State.WAITING
    _handle_output(buffer, *_output(3, "NDextra", "third"))
    stopped_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.STOPPED
    assert stopped_tokens == [("bE" if include else "b", "second"), ("ND" if include else "", "third")]
    assert released_tokens == [("aE", "first")]


def test_none_token_resets_text_matching_without_simulated_flag():
    buffer = StopStrMatchBuffer(_http_manager(), _params(), [])
    _handle_output(buffer, *_output(10, "helloE", "first"))
    first_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.WAITING
    _handle_output(buffer, *_output(None, "N", "marker"))
    marker_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.SCANNING
    _handle_output(buffer, *_output(11, "D", "last"))
    last_tokens = _drain_ready_tokens(buffer)
    assert first_tokens + marker_tokens + last_tokens == [("helloE", "first"), ("N", "marker"), ("D", "last")]
    _handle_output(buffer, *_output(10, "E", "new first"))
    assert buffer.state is buffer.State.WAITING
    _handle_output(buffer, *_output(11, "ND", "new last"))
    assert buffer.ready_token_list[-1][3].is_stopped()


@pytest.mark.parametrize(
    "token_id,finish",
    [(None, FinishStatus.NO_FINISH), (1001, FinishStatus.FINISHED_ERROR), (1001, FinishStatus.FINISHED_ABORTED)],
)
def test_non_real_token_does_not_leave_reasoning(token_id, finish):
    buffer = _stop_buffer()
    _handle_output(buffer, *_output(token_id, "</think>", "marker", finish=finish))
    marker_tokens = _drain_ready_tokens(buffer)
    assert marker_tokens == [("</think>", "marker")]
    assert buffer._reasoning_state.in_reasoning
    if finish == FinishStatus.NO_FINISH:
        _handle_output(buffer, *_output(10, "END", "thought"))
        thought_tokens = _drain_ready_tokens(buffer)
        assert thought_tokens == [("END", "thought")]
    else:
        assert buffer.state is buffer.State.STOPPED


@pytest.mark.parametrize("include", [False, True])
def test_unmatched_tail_flushes(include):
    buffer = StopStrMatchBuffer(_http_manager(), _params(include=include), [])
    _handle_output(buffer, *_output(1, "helloE", 1))
    assert not buffer.ready_token_list
    assert buffer.state is buffer.State.WAITING
    _handle_output(buffer, *_output(2, "N", 2, finish=FinishStatus.FINISHED_LENGTH))
    ready_tokens = _drain_ready_tokens(buffer)
    assert ready_tokens == [("helloE", 1), ("N", 2)]
    assert buffer.state is buffer.State.STOPPED


def test_no_string_stops_emit_immediately():
    buffer = StopStrMatchBuffer(_http_manager(), _params([[10, 11]]), [])
    _handle_output(buffer, *_output(1, "hello", 1))
    ready_tokens = _drain_ready_tokens(buffer)
    assert ready_tokens == [("hello", 1)]


@pytest.mark.parametrize("include", [False, True])
def test_string_stop_matches_reference_for_random_token_boundaries(include):
    async def run():
        rng = random.Random(20261011)
        for _ in range(300):
            stops = ["".join(rng.choices("ab结束", k=rng.randint(1, 5))) for _ in range(rng.randint(1, 4))]
            chunks = ["".join(rng.choices("ab结束", k=rng.randint(0, 4))) for _ in range(20)]
            manager = _http_manager()
            manager.stop_request = AsyncMock()
            buffer = StopStrMatchBuffer(manager, _params(stops, include), [])
            text = ""
            expected_text = None
            expected_count = len(chunks)
            outputs = []
            for index, chunk in enumerate(chunks):
                text += chunk
                if expected_text is None:
                    hits = [(text.find(stop), -len(stop), stop) for stop in stops if stop in text]
                    if hits:
                        position, _, stop = min(hits)
                        expected_text = text[: position + (len(stop) if include else 0)]
                        expected_count = index + 1
                finish = FinishStatus.FINISHED_LENGTH if index == len(chunks) - 1 else FinishStatus.NO_FINISH
                await buffer.handle(*_output(index, chunk, index, finish))
                outputs.extend(_drain_ready_tokens(buffer))
            assert "".join(chunk for chunk, _ in outputs) == (text if expected_text is None else expected_text)
            assert [payload for _, payload in outputs] == list(range(expected_count))
            assert buffer.state is buffer.State.STOPPED
            assert manager.stop_request.await_count == int(expected_text is not None and expected_count < len(chunks))

    asyncio.run(run())


@pytest.mark.parametrize("mode", [StopStrMatchMode.ALL_TEXT, StopStrMatchMode.CONTENT_ONLY])
def test_token_id_stop_is_wrapped_without_text_buffering_even_in_reasoning(mode):
    params = _params([[10, 11]])
    params.stop_str_match_mode.set_mode(mode)
    params._initial_reasoning_state = 1
    manager = _http_manager()
    manager.args.reasoning_parser = "qwen3"
    manager.stop_request = AsyncMock()
    manager.tokenizer.encode = Mock(side_effect=AssertionError("Token ID stops must not encode the prompt"))

    async def raw_results():
        yield 80, "first", {"id": 10}, FinishStatus()
        yield 80, "second", {"id": 11}, FinishStatus()
        yield 80, "late", {"id": 12}, FinishStatus(FinishStatus.FINISHED_LENGTH)

    async def run():
        generator = raw_results()
        wrapped = wrap_stop_match_stream(manager, generator, params, "prompt")
        assert wrapped is not generator
        return [result async for result in wrapped]

    results = asyncio.run(run())
    assert "".join(result[1] for result in results) == "firstsecond"
    assert results[-1][3].is_stopped()
    manager.tokenizer.encode.assert_not_called()
    manager.stop_request.assert_awaited_once_with(80)


@pytest.mark.parametrize(
    "stop,chunks,expected_text",
    [
        ("END", [(1, "E"), (2, "E"), (3, "ND")], "E"),
    ],
)
def test_waiting_tracks_overlapping_stop_prefixes(stop, chunks, expected_text):
    buffer = StopStrMatchBuffer(_http_manager(), _params(stop), [])
    outputs = []
    for index, (token_id, text) in enumerate(chunks):
        _handle_output(buffer, *_output(token_id, text, index))
        ready_tokens = _drain_ready_tokens(buffer)
        outputs.extend(ready_tokens)
        if index < len(chunks) - 1:
            assert buffer.state is buffer.State.WAITING
        else:
            assert buffer.state is buffer.State.STOPPED
            assert buffer.state is buffer.State.STOPPED
    assert "".join(text for text, _ in outputs) == expected_text
    assert [payload for _, payload in outputs] == list(range(len(chunks)))


@pytest.mark.parametrize("mode", [StopStrMatchMode.ALL_TEXT, StopStrMatchMode.CONTENT_ONLY])
def test_token_id_stop_does_not_enter_text_waiting(mode):
    params = _params([[10, 11]])
    params.stop_str_match_mode.set_mode(mode)
    buffer = StopStrMatchBuffer(_http_manager(), params, [])
    _handle_output(buffer, *_output(10, "first", 1))
    first_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.SCANNING
    _handle_output(buffer, *_output(11, "last", 2, finish=FinishStatus.FINISHED_LENGTH))
    last_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.STOPPED
    assert first_tokens + last_tokens == [("first", 1), ("last", 2)]


@pytest.mark.parametrize("stop_in_reasoning,stop_at", [(False, 3), (True, 0)])
@pytest.mark.parametrize("hidden_delimiter", [False, True])
def test_reasoning_stop_matches_only_selected_scope(stop_in_reasoning, stop_at, hidden_delimiter):
    buffer = _stop_buffer(stop_in_reasoning)
    chunks = [(10, "END"), (1001, "" if hidden_delimiter else "</think>"), (11, "answerE"), (12, "NDextra")]
    results = []
    for index, (token_id, text) in enumerate(chunks):
        _handle_output(buffer, *_output(token_id, text, None))
        ready_tokens = _drain_ready_tokens(buffer)
        results.extend(ready_tokens)
        assert (buffer.state is buffer.State.STOPPED) is (index == stop_at)
        if buffer.state is buffer.State.STOPPED:
            break
    assert "".join(text for text, _ in results) == (
        "END" + ("" if hidden_delimiter else "</think>") + "answer" if stop_at == 3 else ""
    )


@pytest.mark.parametrize("include", [False, True])
def test_reasoning_eos_preserves_ignored_stop_text(include):
    buffer = _stop_buffer(include=include)
    _handle_output(buffer, *_output(10, "reasoning END", None))
    ready_tokens = _drain_ready_tokens(buffer)
    assert ready_tokens == [("reasoning END", None)]


@pytest.mark.parametrize("include", [False, True])
def test_scoped_match_filters_content_and_keeps_reasoning(include):
    buffer = _stop_buffer(include=include)
    tokens = []
    for token_id, text in [(10, "END"), (1001, "</think>"), (11, "answerENDextra")]:
        _handle_output(buffer, *_output(token_id, text, None))
        tokens.extend(_drain_ready_tokens(buffer))
    assert "".join(text for text, _ in tokens) == ("END</think>answer" + ("END" if include else ""))


def test_stop_cannot_cross_reasoning_content_boundary():
    buffer = _stop_buffer()
    tokens = []
    for token_id, text in [(1, "E"), (1001, ""), (2, "ND")]:
        _handle_output(buffer, *_output(token_id, text, None))
        assert buffer.state is buffer.State.SCANNING
        tokens.extend(_drain_ready_tokens(buffer))
    assert "".join(text for text, _ in tokens) == "END"


def test_entering_reasoning_releases_and_resets_pending_string_stop():
    buffer = _stop_buffer(force=0)
    _handle_output(buffer, *_output(10, "helloE", "first"))
    first_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.WAITING
    _handle_output(buffer, *_output(1000, "", "start"))
    start_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.SCANNING
    assert first_tokens + start_tokens == [("helloE", "first"), ("", "start")]
    _handle_output(buffer, *_output(11, "ND", "thought"))
    thought_tokens = _drain_ready_tokens(buffer)
    assert thought_tokens == [("ND", "thought")]
    _handle_output(buffer, *_output(1001, "", "end"))
    assert not buffer._reasoning_state.in_reasoning
    _handle_output(buffer, *_output(10, "E", "answer first"))
    _handle_output(buffer, *_output(11, "ND", "answer last"))
    assert buffer.ready_token_list[-1][3].is_stopped()


def test_text_buffer_does_not_match_token_id_stops_in_any_phase():
    buffer = _stop_buffer(stop=[[10, 11]])
    for token_id in [10, 11, 1001, 10, 11]:
        _handle_output(buffer, *_output(token_id, "", None))
        assert buffer.state is buffer.State.SCANNING
    assert all(not output[3].is_finished() for output in buffer.ready_token_list)


def test_pd_first_token_updates_reasoning_state():
    buffer = _stop_buffer()
    _handle_output(buffer, *_output(1001, "</think>", None))
    assert not buffer._reasoning_state.in_reasoning
    _handle_output(buffer, *_output(10, "END", None))
    assert buffer.ready_token_list[-1][3].is_stopped()


def test_prompt_start_marker_prevents_stop_inside_reasoning():
    manager = _http_manager()
    manager.args.reasoning_parser = "qwen3"
    params = _params(force=0)
    params.stop_str_match_mode.set_mode(StopStrMatchMode.CONTENT_ONLY)
    buffer = StopStrMatchBuffer(manager, params, [1, 1000])
    _handle_output(buffer, *_output(10, "END", "thought"))
    thought_tokens = _drain_ready_tokens(buffer)
    assert thought_tokens == [("END", "thought")]
    _handle_output(buffer, *_output(1001, "", "end"))
    assert _drain_ready_tokens(buffer) == [("", "end")]
    _handle_output(buffer, *_output(11, "answerEND", "answer"))
    answer_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.STOPPED
    assert answer_tokens == [("answer", "answer")]


def test_end_marker_outside_reasoning_can_match_stop():
    buffer = _stop_buffer(stop="</think>", force=0)
    payload = object()
    _handle_output(buffer, *_output(1001, "</think>", payload))
    ready_tokens = _drain_ready_tokens(buffer)
    assert buffer.state is buffer.State.STOPPED
    assert ready_tokens == [("", payload)]


def test_pd_capacity_marker_preserves_reasoning_state():
    buffer = _stop_buffer()
    _handle_output(buffer, *_output(None, "", None))
    # 模拟 token 的 </think> 不结束思考，后续 END 仍属于 reasoning。
    _handle_output(buffer, *_output(10, "END", None))
    assert buffer._reasoning_state.in_reasoning
    assert buffer.state is buffer.State.SCANNING
    assert _drain_ready_tokens(buffer) == [("", None), ("END", None)]


@pytest.mark.parametrize(
    "finish",
    [
        FinishStatus.FINISHED_LENGTH,
        FinishStatus.FINISHED_STOP,
        FinishStatus.FINISHED_ERROR,
        FinishStatus.FINISHED_ABORTED,
        FinishStatus.FINISHED_PD_DECODE_CAPACITY,
    ],
)
def test_detokenization_forwards_raw_tokens_before_release(monkeypatch, finish):
    tokens = list(range(10, 30))
    params = _params()
    params.stop_str_match_mode.set_mode(StopStrMatchMode.ALL_TEXT)
    decode = DecodeReq(_req(tokens, finish, params))
    simulated = finish in (
        FinishStatus.FINISHED_ERROR,
        FinishStatus.FINISHED_ABORTED,
        FinishStatus.FINISHED_PD_DECODE_CAPACITY,
    )
    chunks = {tokens[-2]: "helloE", tokens[-1]: "" if simulated else "NDextra"}
    monkeypatch.setattr(
        "lightllm.server.detokenization.manager.decode_token",
        lambda tokenizer, req, token_id, eos: chunks.get(token_id, ""),
    )
    manager = object.__new__(DeTokenizationManager)
    manager.req_id_to_out = {80: decode}
    manager.tokenizer = _ReasoningTokenizer()
    manager.eos_id = []
    manager.all_special_ids = set()
    manager.pub_to_httpserver = SimpleNamespace(send_pyobj=lambda *a, **kw: None)
    manager.shm_req_manager = SimpleNamespace(put_back_req_obj=lambda req: None)
    results = []
    for _ in range(100):
        manager.gen_token_out()
        if decode.out_queue_is_full() and decode.need_detoken():
            assert not decode.req.can_released_mark
        while not decode.req.out_tokens_queue.is_empty():
            results.append(decode.req.out_tokens_queue.pop())
        if decode.req.can_released_mark:
            break
    assert decode.req.can_released_mark
    assert "".join(text for text, *_ in results) == ("helloE" if simulated else "helloENDextra")
    assert [count for *_, count in results] == list(range(1, len(tokens) + 1))


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("empty_tokens", [2, 35])
def test_http_stop_drains_detokenization_before_release(monkeypatch, include, empty_tokens):
    async def run():
        tokens = list(range(10, 13 + empty_tokens))
        params = _params(include=include)
        params.stop_str_match_mode.set_mode(StopStrMatchMode.ALL_TEXT)
        req = _req(tokens, params=params)
        decode = DecodeReq(req)
        chunks = {tokens[0]: "helloE", tokens[-2]: "NDextra"}
        monkeypatch.setattr(
            "lightllm.server.detokenization.manager.decode_token",
            lambda tokenizer, req, token_id, eos: chunks.get(token_id, ""),
        )
        manager = DeTokenizationManager.__new__(DeTokenizationManager)
        manager.req_id_to_out = {80: decode}
        manager.tokenizer = _ReasoningTokenizer()
        manager.eos_id = []
        manager.all_special_ids = set()
        manager.pub_to_httpserver = SimpleNamespace(send_pyobj=lambda *a, **kw: None)
        manager.shm_req_manager = SimpleNamespace(put_back_req_obj=lambda req: None)
        http_manager = _http_manager()

        async def raw_results():
            for _ in tokens:
                if req.is_aborted:
                    req.finish_status.set_status(FinishStatus.FINISHED_ABORTED)
                    req.finish_token_index = req.input_len + len(tokens) - 1
                manager.gen_token_out()
                for result in await _read_http_tokens(http_manager, req):
                    yield result
                if req.can_released_mark:
                    break

        results = [result async for result in _match_http_stream(http_manager, raw_results(), params, [1, 2], [req])]
        assert req.can_released_mark
        assert req.is_aborted
        assert "".join(text for _, text, _, _ in results) == "hello" + ("END" if include else "")
        assert [metadata["id"] for _, _, metadata, _ in results] == tokens[:-1]
        assert all(not status.is_finished() for _, _, _, status in results[:-1])
        assert results[-1][3].is_stopped()
        req.merge_final_token_metadata.assert_awaited_once()

    asyncio.run(run())


def test_http_uses_request_finish_index():
    async def run():
        manager = _http_manager()
        manager.pd_mode = SimpleNamespace(is_normal=lambda: False)
        req = _req([10, 11, 12], FinishStatus.FINISHED_ABORTED)
        req.out_tokens_queue.push("hello", 2, False, 1)
        first = await _read_http_tokens(manager, req)
        req.out_tokens_queue.push("", 4, False, 3)
        last = await _read_http_tokens(manager, req)
        assert first[0][1] == "hello"
        assert not first[0][3].is_finished()
        assert last[0][2]["id"] is None
        assert last[0][3].status == FinishStatus.FINISHED_ABORTED
        req.merge_final_token_metadata.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("include", [False, True])
def test_http_stop_aborts_only_matching_choice_and_drains_late_tokens(include):
    async def run():
        manager = _http_manager()
        params = _params(include=include)
        params.n = params.best_of = 2
        req = _req([10, 11, 12, 13], params=params)
        sibling = _req([50, 51], FinishStatus.FINISHED_LENGTH, params, request_id=81)
        drained = []

        async def raw_results():
            for target, index, text in [
                (req, 0, "helloE"),
                (sibling, 0, "otherE"),
                (req, 1, "NDextra"),
                (req, 2, "late"),
                (sibling, 1, "X"),
                (req, 3, ""),
            ]:
                if target is req and index >= 2:
                    assert req.is_aborted
                    assert not sibling.is_aborted
                if target is req and index == 3:
                    req.finish_status.set_status(FinishStatus.FINISHED_ABORTED)
                    req.finish_token_index = 5
                    req.can_released_mark = True
                target.out_tokens_queue.push(text, 2 + index, False, index + 1)
                drained.append((target.request_id, index))
                for result in await _read_http_tokens(manager, target):
                    yield result

        results = [
            result async for result in _match_http_stream(manager, raw_results(), params, [1, 2], [req, sibling])
        ]
        assert len(drained) == 6
        assert req.out_tokens_queue.is_empty()
        assert "".join(r[1] for r in results if r[0] == 80) == "hello" + ("END" if include else "")
        assert "".join(r[1] for r in results if r[0] == 81) == "otherEX"
        assert [r[2]["id"] for r in results if r[0] == 80] == [10, 11]
        assert {r[0]: r[3].get_finish_reason() for r in results if r[3].is_finished()} == {80: "stop", 81: "length"}
        req.merge_final_token_metadata.assert_awaited_once()
        sibling.merge_final_token_metadata.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize(
    "finish",
    [
        FinishStatus.FINISHED_STOP,
        FinishStatus.FINISHED_LENGTH,
        FinishStatus.FINISHED_ERROR,
        FinishStatus.FINISHED_ABORTED,
    ],
)
def test_http_finish_flushes_unmatched_prefix_and_preserves_final_metadata(finish):
    async def run():
        manager = _http_manager()
        params = _params()
        req = _req([10, 11], finish, params)
        req.merge_final_token_metadata.side_effect = lambda metadata, *args, **kwargs: metadata.update(
            prompt_token_ids=[1, 2]
        )

        async def raw_results():
            for index, text in enumerate(["helloE", ""]):
                req.out_tokens_queue.push(text, 2 + index, False, index + 1)
                for result in await _read_http_tokens(manager, req):
                    yield result

        results = [result async for result in _match_http_stream(manager, raw_results(), params, [1, 2], [req])]
        assert "".join(r[1] for r in results) == "helloE"
        assert results[-1][2]["prompt_token_ids"] == [1, 2]
        assert results[-1][3].status == finish
        assert results[-1][2]["id"] == (None if finish == FinishStatus.FINISHED_ABORTED else 11)
        assert not req.is_aborted

    asyncio.run(run())


@pytest.mark.parametrize("stop_in_reasoning", [False, True])
@pytest.mark.parametrize("token_stop", [False, True])
def test_http_scopes_stop_matching_and_preserves_hidden_reasoning_marker(stop_in_reasoning, token_stop):
    async def run():
        manager = _http_manager()
        manager.args.reasoning_parser = "qwen3"
        params = _params([[10, 11]] if token_stop else "END", force=1)
        params.stop_str_match_mode.set_mode(
            StopStrMatchMode.ALL_TEXT if stop_in_reasoning else StopStrMatchMode.CONTENT_ONLY
        )
        tokens = [10, 11, 1001, 10, 11]
        req = _req(tokens, params=params)
        infer_req = infer_batch.InferReq.__new__(infer_batch.InferReq)
        infer_req.stop_sequences = params.stop_sequences.to_token_ids_list()
        infer_req.shm_req = req
        infer_req.sampling_param = SimpleNamespace(shm_param=params)
        infer_req.finish_status = req.finish_status

        async def raw_results():
            for index, text in enumerate(["E", "ND", "", "answerE", "NDextra"]):
                infer_req.update_finish_status([], output_len=index + 1)
                if req.finish_status.is_finished():
                    req.finish_token_index = req.input_len + index
                req.out_tokens_queue.push(text, 2 + index, False, index + 1)
                for result in await _read_http_tokens(manager, req):
                    yield result
                if req.finish_status.is_finished():
                    break

        results = [result async for result in _match_http_stream(manager, raw_results(), params, [1, 2], [req])]
        expected = "END" if token_stop else ("" if stop_in_reasoning else "ENDanswer")
        assert "".join(r[1] for r in results) == expected
        assert [r[2]["id"] for r in results] == tokens[: 2 if token_stop or stop_in_reasoning else 5]
        assert results[-1][3].is_stopped()

    asyncio.run(run())


def test_stop_groups_can_separate_explicit_token_ids_from_strings():
    params = _params(["END", [10, 11]])
    assert params.stop_sequences.to_token_ids_list() == [[10, 11]]
    assert params.stop_sequences.to_stop_str_list() == ["END"]
    assert params.to_dict()["stop_sequences"] == ["END", [10, 11]]


@pytest.mark.parametrize("mode", [StopStrMatchMode.ALL_TEXT, StopStrMatchMode.CONTENT_ONLY])
def test_string_stop_is_matched_only_after_detokenization(mode):
    params = _params()
    params.stop_str_match_mode.set_mode(mode)
    tokens = [ord(c) for c in "END"]
    shm_req = _req(tokens, params=params)
    req = infer_batch.InferReq.__new__(infer_batch.InferReq)
    req.stop_sequences = params.stop_sequences.to_token_ids_list()
    req.shm_req = shm_req
    req.sampling_param = SimpleNamespace(shm_param=params)
    req.finish_status = FinishStatus()
    req.update_finish_status([], output_len=len(tokens))
    assert not req.finish_status.is_finished()

    decode = DecodeReq(shm_req)
    assert decode.need_detoken()
    assert not decode.can_set_release_mark()


@pytest.mark.parametrize("mode", [StopStrMatchMode.ALL_TEXT, StopStrMatchMode.CONTENT_ONLY])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize(
    "stop_ids, output_ids, matched",
    [
        ([10], [10], True),
        ([10, 11], [10, 11], True),
        ([10, 11], [10], False),
        ([10, 11], [10, 12, 11], False),
        ([10, 10, 11], [10, 10, 10, 11], True),
    ],
)
def test_inference_token_stop_is_independent_of_string_mode_and_reasoning(
    mode, reasoning, stop_ids, output_ids, matched
):
    req = infer_batch.InferReq.__new__(infer_batch.InferReq)
    params = _params(["END", stop_ids], force=int(reasoning))
    params.stop_str_match_mode.set_mode(mode)
    req.stop_sequences = params.stop_sequences.to_token_ids_list()
    req.shm_req = SimpleNamespace(input_len=1, shm_prompt_ids=SimpleNamespace(arr=np.array([1, *output_ids])))
    req.sampling_param = SimpleNamespace(shm_param=params)
    req.finish_status = FinishStatus()
    req.update_finish_status([], output_len=len(output_ids))
    assert req.finish_status.is_stopped() is matched

    # EOS and the output length limit remain independent of user stop sequences.
    req.finish_status = FinishStatus()
    req.update_finish_status([output_ids[-1]], output_len=len(output_ids))
    assert req.finish_status.is_stopped() is True
    req.finish_status = FinishStatus()
    req.stop_sequences = []
    req.sampling_param.shm_param.max_new_tokens = 1
    req.update_finish_status([], output_len=1)
    assert req.finish_status.is_finished_length() is True


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("last_status", [FinishStatus.NO_FINISH, FinishStatus.FINISHED_LENGTH])
@pytest.mark.parametrize(
    "stops, output_ids, match_index",
    [
        ([[10]], [10, 20], 0),
        ([[10, 11]], [10, 11, 20], 1),
        ([[10, 10, 11]], [10, 10, 10, 11, 20], 3),
        ([[10, 11], [11, 12]], [11, 12, 20], 1),
        ([[10, 11]], [10, 12, 11], None),
    ],
)
def test_token_id_buffer_preserves_tokens_and_only_corrects_matched_finish(
    include, last_status, stops, output_ids, match_index
):
    async def run():
        manager = _http_manager()
        manager.stop_request = AsyncMock()
        buffer = StopTokenIdsMatchBuffer(manager, _params(stops, include))
        results = []
        raw_outputs = []
        for index, token_id in enumerate(output_ids):
            status = last_status if index == match_index or index == len(output_ids) - 1 else FinishStatus.NO_FINISH
            output = _output(token_id, "" if token_id == 10 else str(token_id), index, status)
            raw_outputs.append(output)
            await buffer.handle(*output)
            while buffer.ready_token_list:
                results.append(buffer.ready_token_list.popleft())
        expected = raw_outputs if match_index is None else raw_outputs[: match_index + 1]
        assert [r[:3] for r in results] == [r[:3] for r in expected]
        assert all(result[2] is raw[2] for result, raw in zip(results, expected))
        if match_index is None:
            assert results[-1][3].status == last_status
            manager.stop_request.assert_not_awaited()
        else:
            assert results[-1][3].is_stopped()
            assert raw_outputs[match_index][3].status == last_status
            if last_status == FinishStatus.NO_FINISH:
                manager.stop_request.assert_awaited_once_with(80)
            else:
                manager.stop_request.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("last_status", [FinishStatus.FINISHED_ERROR, FinishStatus.FINISHED_LENGTH])
def test_token_id_stream_preserves_error_and_simulated_final_output(last_status):
    manager = _http_manager()
    manager.stop_request = AsyncMock()
    params = _params([[10, 11]])

    async def raw_results():
        yield _output(10, "first", 0)
        yield _output(11 if last_status == FinishStatus.FINISHED_ERROR else None, "final", 1, last_status)
        yield _output(11, "late", 2)

    async def run():
        return [result async for result in match_stop_token_ids_stream(manager, raw_results(), params)]

    results = asyncio.run(run())
    assert [r[1] for r in results] == ["first", "final"]
    assert results[-1][3].status == last_status
    manager.stop_request.assert_not_awaited()


def test_token_id_stream_keeps_interleaved_choices_independent():
    manager = _http_manager()
    manager.stop_request = AsyncMock()
    params = _params([[10, 11]])
    params.n = params.best_of = 2

    async def raw_results():
        for req_id, token_id in [(80, 10), (81, 11), (80, 11), (80, 20), (81, 10), (81, 11), (81, 20)]:
            yield req_id, str(token_id), {"id": token_id}, FinishStatus()

    async def run():
        return [result async for result in match_stop_token_ids_stream(manager, raw_results(), params)]

    results = asyncio.run(run())
    assert [r[2]["id"] for r in results if r[0] == 80] == [10, 11]
    assert [r[2]["id"] for r in results if r[0] == 81] == [11, 10, 11]
    assert [r[0] for r in results if r[3].is_stopped()] == [80, 81]
    assert [call.args for call in manager.stop_request.await_args_list] == [(80,), (81,)]


def test_token_id_stream_does_not_match_prompt_ids():
    manager = _http_manager()
    manager.stop_request = AsyncMock()
    params = _params([[10, 11]])

    async def raw_results():
        yield 80, "answer", {"id": 11, "prompt_token_ids": [10]}, FinishStatus(FinishStatus.FINISHED_LENGTH)

    async def run():
        return [result async for result in wrap_stop_match_stream(manager, raw_results(), params, [10])]

    results = asyncio.run(run())
    assert results[0][3].is_finished_length()
    manager.stop_request.assert_not_awaited()


def test_token_id_matching_runs_before_string_matching():
    manager = _http_manager()
    manager.stop_request = AsyncMock()
    params = _params(["END", [10, 11]])

    async def raw_results():
        yield _output(10, "helloE", 0)
        yield _output(11, "N", 1)
        yield _output(12, "Dextra", 2, FinishStatus.FINISHED_LENGTH)

    async def run():
        # prompt 的 ID 即使是 stop 前缀，也不进入 token ID 匹配历史。
        return [result async for result in wrap_stop_match_stream(manager, raw_results(), params, [10])]

    results = asyncio.run(run())
    assert "".join(r[1] for r in results) == "helloEN"
    assert [r[2]["id"] for r in results] == [10, 11]
    assert results[-1][3].is_stopped()
    manager.stop_request.assert_awaited_once_with(80)


@pytest.mark.parametrize("with_stop", [None, "END", [[10, 11]], ["END", [10, 11]]])
def test_stop_stream_close_closes_raw_generator(with_stop):
    params = _params(stop=with_stop)
    closed = []

    async def raw_results():
        try:
            yield 80, "hello", {"id": 10}, FinishStatus()
            pytest.fail("Closing the wrapper must not consume another raw token")
        finally:
            closed.append(True)

    async def run():
        manager = SimpleNamespace(
            args=SimpleNamespace(enable_rl=False, enable_return_routed_experts=False, reasoning_parser=None),
            tokenizer=_ReasoningTokenizer(),
            stop_request=AsyncMock(),
        )
        raw_generator = raw_results()
        output = wrap_stop_match_stream(manager, raw_generator, params, "prompt")
        assert (output is raw_generator) is (not with_stop)
        result = await output.__anext__()
        assert result[1] == "hello"
        await output.aclose()

    asyncio.run(run())
    assert closed == [True]


@pytest.mark.parametrize("node_finished", [False, True])
def test_stop_stream_calls_manager_without_mutating_raw_finish(node_finished):
    params = _params()
    raw_finish = FinishStatus(FinishStatus.FINISHED_LENGTH if node_finished else FinishStatus.NO_FINISH)
    manager = _http_manager()
    manager.stop_request = AsyncMock()

    async def raw_results():
        yield 80, "helloENDextra", {"id": 10}, raw_finish
        yield 80, "late", {"id": 11}, FinishStatus(FinishStatus.FINISHED_LENGTH)

    async def run():
        return [result async for result in match_stop_str_stream(manager, raw_results(), params, "prompt")]

    results = asyncio.run(run())
    assert results[0][1] == "hello"
    assert results[0][3].is_stopped()
    assert raw_finish.status == (FinishStatus.FINISHED_LENGTH if node_finished else FinishStatus.NO_FINISH)
    if node_finished:
        manager.stop_request.assert_not_awaited()
    else:
        manager.stop_request.assert_awaited_once_with(80)


@pytest.mark.parametrize("include", [False, True])
def test_http_stream_discards_late_tokens_in_same_full_queue_batch(include):
    async def run():
        manager = _http_manager()
        params = _params(include=include)
        req = _req(list(range(10, 20)), params=params)
        index = 0
        while not req.out_tokens_queue.is_full():
            chunks = ["helloE", "N", "Dextra"]
            req.out_tokens_queue.push(chunks[index] if index < len(chunks) else "late", 2 + index, False, index + 1)
            index += 1

        async def raw_results():
            for result in await _read_http_tokens(manager, req):
                yield result

        results = [result async for result in _match_http_stream(manager, raw_results(), params, [1, 2], [req])]
        assert req.out_tokens_queue.is_empty()
        assert req.is_aborted
        assert "".join(r[1] for r in results) == "hello" + ("END" if include else "")
        assert [r[2]["id"] for r in results] == [10, 11, 12]
        assert results[-1][3].is_stopped()

    asyncio.run(run())


def test_stop_stream_uses_prompt_tail_for_all_choices_and_ignores_metadata():
    params = _params(force=0)
    params.n = params.best_of = 2
    params.stop_str_match_mode.set_mode(StopStrMatchMode.CONTENT_ONLY)
    manager = _http_manager()
    manager.args.reasoning_parser = "qwen3"
    manager.stop_request = AsyncMock()
    manager.tokenizer.encode = Mock(wraps=manager.tokenizer.encode)

    async def raw_results():
        yield 80, "END", {"id": 10, "prompt_token_ids": [1, 1001]}, FinishStatus()
        yield 81, "END", {"id": 10}, FinishStatus()
        for req_id in (80, 81):
            yield req_id, "", {"id": 1001}, FinishStatus()
            yield req_id, "answerENDextra", {"id": 11}, FinishStatus()

    async def run():
        return [result async for result in match_stop_str_stream(manager, raw_results(), params, [1] * 4096 + [1000])]

    results = asyncio.run(run())
    for req_id in (80, 81):
        assert "".join(result[1] for result in results if result[0] == req_id) == "ENDanswer"
    assert [call.args for call in manager.stop_request.await_args_list] == [(80,), (81,)]
    assert [call.args for call in manager.tokenizer.encode.call_args_list] == [("<think>",), ("</think>",)] * 2
    assert results[0][2]["prompt_token_ids"] == [1, 1001]


@pytest.mark.parametrize("prompt", ["prompt", "x" * 4096 + "prompt", [1, 2], [1] * 4096 + [2]])
@pytest.mark.parametrize("prompt_metadata", [{}, {"prompt_token_ids": None}, {"prompt_token_ids": [1000]}])
@pytest.mark.parametrize(
    "mode, parser",
    [
        (StopStrMatchMode.ALL_TEXT, "qwen3"),
        (StopStrMatchMode.CONTENT_ONLY, None),
        (StopStrMatchMode.CONTENT_ONLY, "qwen3"),
    ],
)
def test_stop_stream_encodes_prompt_tail_independent_of_mode_and_metadata(prompt, prompt_metadata, mode, parser):
    params = _params()
    params.stop_str_match_mode.set_mode(mode)
    manager = _http_manager()
    manager.args.reasoning_parser = parser
    manager.stop_request = AsyncMock()
    manager.tokenizer.encode = Mock(wraps=manager.tokenizer.encode)

    async def raw_results():
        yield 80, "helloENDextra", {"id": 10, **prompt_metadata}, FinishStatus()

    async def run():
        return [result async for result in match_stop_str_stream(manager, raw_results(), params, prompt)]

    results = asyncio.run(run())
    assert results[0][1] == "hello"
    assert results[0][3].is_stopped()
    prompt_calls = [
        call for call in manager.tokenizer.encode.call_args_list if call.args[0] not in ("<think>", "</think>")
    ]
    assert len(prompt_calls) == int(isinstance(prompt, str))
    if isinstance(prompt, str):
        assert prompt_calls[0].args == (prompt[-2048:],)
        assert prompt_calls[0].kwargs == {"add_special_tokens": False}


def test_stop_stream_keeps_reasoning_state_independent_for_interleaved_choices():
    params = _params(force=0)
    params.n = params.best_of = 2
    params.stop_str_match_mode.set_mode(StopStrMatchMode.CONTENT_ONLY)
    manager = _http_manager()
    manager.args.reasoning_parser = "qwen3"
    manager.stop_request = AsyncMock()

    async def raw_results():
        for req_id, token_id, text in [
            (80, 1000, ""),
            (81, 10, "END"),
            (80, 10, "END"),
            (81, 11, "late"),
            (80, 1001, ""),
            (80, 12, "answerENDextra"),
        ]:
            yield req_id, text, {"id": token_id}, FinishStatus()

    async def run():
        return [result async for result in match_stop_str_stream(manager, raw_results(), params, [])]

    results = asyncio.run(run())
    assert "".join(r[1] for r in results if r[0] == 80) == "ENDanswer"
    assert "".join(r[1] for r in results if r[0] == 81) == ""
    assert [call.args for call in manager.stop_request.await_args_list] == [(81,), (80,)]
    assert params._initial_reasoning_state == 0


@pytest.mark.parametrize("first_finished", [False, True])
@pytest.mark.parametrize(
    "late_status", [FinishStatus.NO_FINISH, FinishStatus.FINISHED_LENGTH, FinishStatus.FINISHED_ERROR]
)
def test_stopped_choice_ignores_late_output_before_and_after_sibling_starts(first_finished, late_status):
    params = _params()
    params.n = params.best_of = 2
    manager = _http_manager()
    manager.stop_request = AsyncMock()
    closed = []
    exhausted = []

    async def raw_results():
        try:
            finish = FinishStatus(FinishStatus.FINISHED_LENGTH if first_finished else FinishStatus.NO_FINISH)
            yield 80, "first" if first_finished else "firstENDextra", {"id": 10}, finish
            yield 80, "lateEND", {"id": 11}, FinishStatus(late_status)
            yield 81, "secondE", {"id": 20}, FinishStatus()
            yield 80, "laterEND", {"id": 12}, FinishStatus(late_status)
            yield 81, "NDextra", {"id": 21}, FinishStatus(FinishStatus.FINISHED_LENGTH)
            yield 80, "lastEND", {"id": 13}, FinishStatus(late_status)
            yield 81, "lastEND", {"id": 22}, FinishStatus(late_status)
            exhausted.append(True)
        finally:
            closed.append(True)

    async def run():
        return [result async for result in match_stop_str_stream(manager, raw_results(), params, [])]

    results = asyncio.run(run())
    assert [result[0] for result in results] == [80, 81, 81]
    assert "".join(result[1] for result in results) == "firstsecond"
    assert results[0][3].status == (FinishStatus.FINISHED_LENGTH if first_finished else FinishStatus.FINISHED_STOP)
    assert results[-1][3].is_stopped()
    if first_finished:
        manager.stop_request.assert_not_awaited()
    else:
        manager.stop_request.assert_awaited_once_with(80)
    assert closed == [True]
    assert exhausted == [True]


def test_application_stop_keeps_hit_metadata_and_discards_backend_finish():
    params = _params()
    manager = _http_manager()
    manager.stop_request = AsyncMock()

    async def raw_results():
        yield 80, "helloE", {"id": 10, "logprob": -0.1, "count_output_tokens": 1}, FinishStatus()
        yield 80, "NDextra", {"id": 11, "logprob": -0.2, "count_output_tokens": 2}, FinishStatus()
        yield 80, "late", {"id": 12, "logprob": -0.3, "count_output_tokens": 3}, FinishStatus(
            FinishStatus.FINISHED_LENGTH
        )

    async def run():
        return [result async for result in match_stop_str_stream(manager, raw_results(), params, [])]

    results = asyncio.run(run())
    assert "".join(r[1] for r in results) == "hello"
    assert results[-1][3].is_stopped()
    assert results[-1][2] == {
        "id": 11,
        "logprob": -0.2,
        "count_output_tokens": 2,
    }
    manager.stop_request.assert_awaited_once_with(80)


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("finished", [False, True])
def test_ready_queue_preserves_unconsumed_tokens_after_stop(include, finished):
    buffer = StopStrMatchBuffer(_http_manager(), _params(include=include), [])
    ready_queue = buffer.ready_token_list
    assert isinstance(ready_queue, deque)
    payloads = [object() for _ in range(4)]
    _handle_output(buffer, *_output(1, "one", payloads[0]))
    _handle_output(buffer, *_output(2, "two", payloads[1]))
    _handle_output(buffer, *_output(3, "helloE", payloads[2]))
    assert buffer.state is buffer.State.WAITING
    output = ready_queue.popleft()
    assert output[:3] == (80, "one", {"id": 1, "payload": payloads[0]})
    assert not output[3].is_finished()
    finish = FinishStatus.FINISHED_LENGTH if finished else FinishStatus.NO_FINISH
    _handle_output(buffer, *_output(4, "N" if finished else "NDextra", payloads[3], finish=finish))
    assert ready_queue[-1][3].is_stopped() is (not finished)
    assert buffer.state is buffer.State.STOPPED
    _handle_output(buffer, *_output(5, "late", object()))
    assert buffer.ready_token_list is ready_queue
    assert _drain_ready_tokens(buffer) == [
        ("two", payloads[1]),
        ("helloE" if finished or include else "hello", payloads[2]),
        ("N" if finished else "ND" if include else "", payloads[3]),
    ]
    assert not ready_queue


@pytest.mark.parametrize("matched,finished", [(False, True), (True, False), (True, True)])
def test_stopped_buffer_ignores_later_tokens(matched, finished):
    buffer = _stop_buffer(stop_in_reasoning=True)
    finish = FinishStatus.FINISHED_LENGTH if finished else FinishStatus.NO_FINISH
    _handle_output(buffer, *_output(10, "helloEND" if matched else "helloE", "first", finish=finish))
    first_finish = buffer.ready_token_list[-1][3]
    first_tokens = _drain_ready_tokens(buffer)
    assert first_finish.is_stopped() is matched
    assert "".join(text for text, _ in first_tokens) == ("hello" if matched else "helloE")
    assert buffer.state is buffer.State.STOPPED
    for finished in [False, True]:
        finish = FinishStatus.FINISHED_LENGTH if finished else FinishStatus.NO_FINISH
        _handle_output(buffer, *_output(11, "laterEND", "later", finish=finish))
        ready_tokens = _drain_ready_tokens(buffer)
        assert not ready_tokens
    assert "".join(text for text, _ in first_tokens) == ("hello" if matched else "helloE")


def test_simulated_finish_releases_prefix_and_preserves_payloads():
    buffer = _stop_buffer(stop_in_reasoning=True)
    first_payload, finish_payload = object(), object()
    _handle_output(buffer, *_output(10, "helloE", first_payload))
    assert not buffer.ready_token_list
    _handle_output(buffer, *_output(None, "", finish_payload, finish=FinishStatus.FINISHED_LENGTH))
    ready_tokens = _drain_ready_tokens(buffer)
    assert ready_tokens == [("helloE", first_payload), ("", finish_payload)]
    assert buffer.state is buffer.State.STOPPED
