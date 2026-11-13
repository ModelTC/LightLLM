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
from lightllm.server.httpserver.req_id_generator_server import convert_sub_id_to_group_id


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
        (["abcde", "c"], ["ab", "c"], "ab", "c"),
        (["ab", "abcd"], ["ab"], "", "ab"),
        (["ab", "abcd"], ["abcdextra"], "", "abcd"),
        (["b", "abcd"], ["abcdextra"], "", "abcd"),
        ([".*"], ["hello.", "*extra"], "hello", ".*"),
        (["[END]"], ["hello[EN", "D]extra"], "hello", "[END]"),
        (["\\end"], ["hello\\", "endextra"], "hello", "\\end"),
        (["🙂结束"], ["你好🙂结", "束后缀"], "你好", "🙂结束"),
        (["a" * 254 + "b"], ["hello" + "a" * 253, "a", "", "bextra"], "hello", "a" * 254 + "b"),
        (["结" * 84 + "END"], ["你好" + "结" * 83, "结E", "ND后缀"], "你好", "结" * 84 + "END"),
        (["🙂" * 63 + "END"], ["hello" + "🙂" * 62, "🙂E", "NDextra"], "hello", "🙂" * 63 + "END"),
        (["aaaab", "aaac"], ["aaa", "a", "c"], "a", "aaac"),
        (["\r\n"], ["hello\r", "\nextra"], "hello", "\r\n"),
        (["e\u0301END"], ["helloe", "\u0301EN", "Dextra"], "hello", "e\u0301END"),
        (["(?i)[]+*.^$|{}\\"], ["hello(?i)[]+", "*.^$|{}\\extra"], "hello", "(?i)[]+*.^$|{}\\"),
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
