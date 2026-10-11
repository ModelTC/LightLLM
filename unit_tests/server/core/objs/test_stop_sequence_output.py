import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import numpy as np

from lightllm.server.core.objs import FinishStatus, SamplingParams, Req
from lightllm.server.core.objs.out_token_circlequeue import CircularQueue
from lightllm.server.detokenization.decode_req import DecodeReq
from lightllm.server.detokenization.manager import DeTokenizationManager
from lightllm.server.httpserver.stop_sequence import StopSequenceBuffer
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.router.model_infer import infer_batch


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
        enable_stop_match_in_backend=False,
    )
    return params


def _stop_buffer(stop_in_reasoning=False, stop="END", force=1, parser="qwen3", include=False):
    params = _params(stop, include, force)
    return StopSequenceBuffer(params, _ReasoningTokenizer(), [1, 2], None if stop_in_reasoning else parser)


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
        stop_str_matched=False,
        stop_str_matched_token_index=-1,
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
    manager.args = SimpleNamespace(use_reward_model=False, enable_return_routed_experts=False)
    return manager


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
    buffer = StopSequenceBuffer(_params(stops, include), _ReasoningTokenizer(), [])
    payloads = [object() for _ in chunks]
    results = []
    for index, (text, payload) in enumerate(zip(chunks, payloads)):
        matched = buffer.append(index, text, payload)
        results.extend(buffer.ready_tokens)
        buffer.ready_tokens.clear()
        assert matched is (index == len(chunks) - 1)
    assert "".join(text for text, _ in results) == prefix + (matched_stop if include else "")
    assert [payload for _, payload in results] == payloads
    assert not buffer.output_strs


@pytest.mark.parametrize("include", [False, True])
def test_only_possible_stop_prefix_delays_whole_token(include):
    buffer = StopSequenceBuffer(_params(include=include), _ReasoningTokenizer(), [])
    assert not buffer.append(1, "hello", 1)
    assert list(buffer.ready_tokens) == [("hello", 1)]
    buffer.ready_tokens.clear()
    assert not buffer.append(2, "worldE", 2)
    assert not buffer.ready_tokens
    assert not buffer.append(3, "X", 3)
    assert list(buffer.ready_tokens) == [("worldE", 2), ("X", 3)]


@pytest.mark.parametrize("include", [False, True])
def test_unmatched_tail_flushes(include):
    buffer = StopSequenceBuffer(_params(include=include), _ReasoningTokenizer(), [])
    buffer.append(1, "helloE", 1)
    buffer.append(2, "N", 2)
    assert not buffer.ready_tokens
    buffer.flush()
    assert list(buffer.ready_tokens) == [("helloE", 1), ("N", 2)]


def test_no_string_stops_emit_immediately():
    buffer = StopSequenceBuffer(_params([[10, 11]]), _ReasoningTokenizer(), [])
    assert not buffer.append(1, "hello", 1)
    assert list(buffer.ready_tokens) == [("hello", 1)]


def test_token_stops_can_be_delegated_without_reasoning_parser():
    params = _params([[10, 11]])
    params.enable_stop_match_in_backend = False
    buffer = StopSequenceBuffer(params, _ReasoningTokenizer(), [])
    assert not buffer.append(10, "first", 1)
    assert buffer.append(11, "second", 2)
    assert list(buffer.ready_tokens) == [("first", 1), ("second", 2)]


@pytest.mark.parametrize("stop_in_reasoning,stop_at", [(False, 3), (True, 0)])
@pytest.mark.parametrize("hidden_delimiter", [False, True])
def test_reasoning_stop_matches_only_selected_scope(stop_in_reasoning, stop_at, hidden_delimiter):
    buffer = _stop_buffer(stop_in_reasoning)
    chunks = [(10, "END"), (1001, "" if hidden_delimiter else "</think>"), (11, "answerE"), (12, "NDextra")]
    results = []
    for index, (token_id, text) in enumerate(chunks):
        matched = buffer.append(token_id, text, None)
        results.extend(buffer.ready_tokens)
        buffer.ready_tokens.clear()
        assert matched is (index == stop_at)
        if matched:
            break
    assert "".join(text for text, _ in results) == (
        "END" + ("" if hidden_delimiter else "</think>") + "answer" if stop_at == 3 else ""
    )


@pytest.mark.parametrize("include", [False, True])
def test_reasoning_eos_preserves_ignored_stop_text(include):
    buffer = _stop_buffer(include=include)
    assert not buffer.append(10, "reasoning END", None)
    assert list(buffer.ready_tokens) == [("reasoning END", None)]


@pytest.mark.parametrize("include", [False, True])
def test_scoped_match_filters_content_and_keeps_reasoning(include):
    buffer = _stop_buffer(include=include)
    for token_id, text in [(10, "END"), (1001, "</think>"), (11, "answerENDextra")]:
        buffer.append(token_id, text, None)
    assert "".join(text for text, _ in buffer.ready_tokens) == ("END</think>answer" + ("END" if include else ""))


def test_stop_cannot_cross_reasoning_content_boundary():
    buffer = _stop_buffer()
    for token_id, text in [(1, "E"), (1001, ""), (2, "ND")]:
        assert not buffer.append(token_id, text, None)


def test_token_id_stops_ignore_reasoning_and_match_content():
    buffer = _stop_buffer(stop=[[10, 11]])
    for token_id in [10, 11, 1001, 10]:
        assert not buffer.append(token_id, "", None)
    assert buffer.append(11, "", None)


def test_pd_first_token_updates_reasoning_state():
    buffer = _stop_buffer()
    assert not buffer.append(1001, "</think>", None)
    assert buffer.append(10, "END", None)


def test_pd_capacity_marker_preserves_reasoning_state():
    buffer = _stop_buffer()
    assert not buffer.append(1001, "", None, simulated=True)
    assert buffer.reasoning_state.in_reasoning is True


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
@pytest.mark.parametrize("backend", [False, True])
def test_detokenization_forwards_raw_tokens_before_release(monkeypatch, finish, backend):
    tokens = list(range(10, 30))
    params = _params()
    params.enable_stop_match_in_backend = backend
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
    assert decode.req.stop_str_matched is (backend and finish == FinishStatus.FINISHED_LENGTH)
    assert "".join(text for text, *_ in results) == ("helloE" if simulated else "helloENDextra")
    assert [count for *_, count in results] == list(range(1, len(tokens) + 1))


@pytest.mark.parametrize("backend", [False, True])
def test_stop_mode_preserves_backend_match_and_forwards_raw_tokens(monkeypatch, backend):
    tokens = list(range(10, 20))
    params = _params()
    params.enable_stop_match_in_backend = backend
    req = _req(tokens, FinishStatus.FINISHED_LENGTH, params)
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
    results = []
    for _ in range(100):
        manager.gen_token_out()
        while not req.out_tokens_queue.is_empty():
            results.append(req.out_tokens_queue.pop())
        if req.can_released_mark:
            break
    assert req.can_released_mark
    assert "".join(text for text, *_ in results) == "helloENDextra"
    assert [count for *_, count in results] == list(range(1, len(tokens) + 1 - int(backend)))
    assert req.stop_str_matched_token_index == (2 + len(tokens) - 2 if backend else -1)


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("empty_tokens", [2, 35])
def test_backend_stop_match_is_buffered_and_trimmed_by_http(monkeypatch, include, empty_tokens):
    async def run():
        tokens = list(range(10, 13 + empty_tokens))
        params = _params(include=include)
        params.enable_stop_match_in_backend = True
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
        buffer = StopSequenceBuffer(params, manager.tokenizer, [1, 2])
        results = []
        for _ in tokens:
            if req.is_aborted:
                req.finish_status.set_status(FinishStatus.FINISHED_ABORTED)
                req.finish_token_index = req.input_len + len(tokens) - 1
            manager.gen_token_out()
            results.extend(await http_manager._read_req_tokens(req, buffer))
            if req.can_released_mark:
                break
            # 跨 token 的 stop 前缀不能提前输出，包括其后的连续空文本 token。
            if not buffer.matched:
                assert results == []

        assert req.can_released_mark
        backend_matched = empty_tokens == 2
        assert req.stop_str_matched is backend_matched
        assert req.is_aborted is (not backend_matched)
        assert decode.output_strs == ["helloE", *([""] * empty_tokens), "NDextra"] + ([] if backend_matched else [""])
        assert "".join(text for _, text, _, _ in results) == "hello" + ("END" if include else "")
        assert [metadata["id"] for _, _, metadata, _ in results] == tokens[:-1]
        assert [metadata["count_output_tokens"] for _, _, metadata, _ in results] == list(range(1, len(tokens)))
        assert all(not status.is_finished() for _, _, _, status in results[:-1])
        assert results[-1][3].is_stopped()
        req.merge_final_token_metadata.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("backend", [False, True])
def test_http_uses_legacy_stop_token_index_only_in_backend_mode(backend):
    async def run():
        manager = _http_manager()
        params = _params()
        params.enable_stop_match_in_backend = backend
        req = _req([10, 11, 12], FinishStatus.FINISHED_ABORTED, params)
        req.stop_str_matched = True
        req.stop_str_matched_token_index = 3
        req.out_tokens_queue.push("hello", 2, False, 1)
        first = await manager._read_req_tokens(req)
        req.out_tokens_queue.push("", 3, False, 2)
        last = await manager._read_req_tokens(req)
        assert first[0][1] == "hello"
        assert not first[0][3].is_finished()
        assert last[0][2]["id"] == 11
        assert last[0][3].is_stopped() is backend
        assert req.merge_final_token_metadata.await_count == int(backend)

    asyncio.run(run())


@pytest.mark.parametrize("include", [False, True])
def test_http_stop_aborts_only_matching_choice_and_drains_late_tokens(include):
    async def run():
        manager = _http_manager()
        params = _params(include=include)
        tokens = list(range(10, 45))
        req = _req(tokens, params=params)
        sibling = _req([50], FinishStatus.FINISHED_LENGTH, params, request_id=81)
        buffer = StopSequenceBuffer(params, manager.tokenizer, [1, 2])
        results = []
        match_index = len(tokens) - 3
        for index in range(len(tokens)):
            if index == len(tokens) - 1:
                req.finish_status.set_status(FinishStatus.FINISHED_ABORTED)
                req.finish_token_index = 2 + index
                req.can_released_mark = True
            text = (
                "helloE" if index == 0 else "NDextra" if index == match_index else "late" if index > match_index else ""
            )
            req.out_tokens_queue.push(text, 2 + index, False, index + 1)
            results.extend(await manager._read_req_tokens(req, buffer))
            if index == match_index:
                assert req.is_aborted
                assert not sibling.is_aborted
                assert not Req.can_release(req)

        assert Req.can_release(req)
        assert req.cumlogprob == -0.5 * (match_index + 1)
        assert "".join(text for _, text, _, _ in results) == "hello" + ("END" if include else "")
        assert [metadata["id"] for _, _, metadata, _ in results] == tokens[: match_index + 1]
        assert [metadata["count_output_tokens"] for _, _, metadata, _ in results] == list(range(1, match_index + 2))
        assert all(metadata["logprob"] == -0.5 for _, _, metadata, _ in results)
        assert all(not finish.is_finished() for _, _, _, finish in results[:-1])
        assert results[-1][3].is_stopped()
        req.merge_final_token_metadata.assert_awaited_once()

        sibling.out_tokens_queue.push("otherE", 2, False, 1)
        sibling_buffer = StopSequenceBuffer(params, manager.tokenizer, [1, 2])
        output = await manager._read_req_tokens(sibling, sibling_buffer)
        assert output[0][1] == "otherE"
        assert output[0][3].is_finished_length()
        assert not sibling.is_aborted

    asyncio.run(run())


def test_http_discards_tokens_after_stop_in_same_full_queue_batch():
    async def run():
        manager = _http_manager()
        req = _req(list(range(10, 20)))
        buffer = StopSequenceBuffer(req.sample_params, manager.tokenizer, [1, 2])
        chunks = ["helloE", "N", "Dextra"]
        index = 0
        while not req.out_tokens_queue.is_full():
            req.out_tokens_queue.push(chunks[index] if index < len(chunks) else "late", 2 + index, False, index + 1)
            index += 1
        results = await manager._read_req_tokens(req, buffer)
        assert req.out_tokens_queue.is_empty()
        assert req.is_aborted
        assert "".join(text for _, text, _, _ in results) == "hello"
        assert [metadata["id"] for _, _, metadata, _ in results] == [10, 11, 12]
        assert req.cumlogprob == -1.5
        assert results[-1][3].is_stopped()

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
        buffer = StopSequenceBuffer(params, manager.tokenizer, [1, 2])
        req.merge_final_token_metadata.side_effect = lambda metadata, *args, **kwargs: metadata.update(
            prompt_token_ids=[1, 2]
        )
        req.out_tokens_queue.push("helloE", 2, False, 1)
        assert not await manager._read_req_tokens(req, buffer)
        req.out_tokens_queue.push("", 3, True, 2)
        results = await manager._read_req_tokens(req, buffer)
        assert "".join(text for _, text, _, _ in results) == "helloE"
        assert results[-1][2]["prompt_token_ids"] == [1, 2]
        assert results[-1][3].status == finish
        assert results[-1][2]["id"] == (None if finish == FinishStatus.FINISHED_ABORTED else 11)
        assert not buffer.matched
        assert not req.is_aborted

    asyncio.run(run())


@pytest.mark.parametrize("stop_in_reasoning", [False, True])
@pytest.mark.parametrize("token_stop", [False, True])
def test_http_scopes_stop_matching_and_preserves_hidden_reasoning_marker(stop_in_reasoning, token_stop):
    async def run():
        manager = _http_manager()
        params = _params([[10, 11]] if token_stop else "END", force=1)
        tokens = [10, 11, 1001, 10, 11]
        req = _req(tokens, params=params)
        buffer = StopSequenceBuffer(params, manager.tokenizer, [1, 2], None if stop_in_reasoning else "qwen3")
        results = []
        for index, text in enumerate(["E", "ND", "", "answerE", "NDextra"]):
            req.out_tokens_queue.push(text, 2 + index, False, index + 1)
            results.extend(await manager._read_req_tokens(req, buffer))
            if results and results[-1][3].is_finished():
                break
        if token_stop:
            expected = "END" if stop_in_reasoning else "ENDanswerENDextra"
        else:
            expected = "" if stop_in_reasoning else "ENDanswer"
        assert "".join(text for _, text, _, _ in results) == expected
        assert [metadata["id"] for _, _, metadata, _ in results] == tokens[: 2 if stop_in_reasoning else 5]
        assert results[-1][3].is_stopped()

    asyncio.run(run())


def test_stop_groups_can_separate_explicit_token_ids_from_strings():
    params = _params(["END", [10, 11]])
    assert params.stop_sequences.to_list(include_strings=False) == [[10, 11]]
    assert params.stop_sequences.to_list() == [[ord(c) for c in "END"], [10, 11]]


@pytest.mark.parametrize("match_in_inference", [False, True])
def test_inference_stop_gate_uses_sampling_params(match_in_inference):
    req = infer_batch.InferReq.__new__(infer_batch.InferReq)
    req.stop_sequences = [[10]]
    req.shm_req = SimpleNamespace(input_len=1, shm_prompt_ids=SimpleNamespace(arr=np.array([1, 10])))
    req.sampling_param = SimpleNamespace(
        shm_param=SimpleNamespace(ignore_eos=False, max_new_tokens=100, enable_stop_match_in_backend=match_in_inference)
    )
    req.finish_status = FinishStatus()
    req.update_finish_status([], output_len=1)
    assert req.finish_status.is_stopped() is match_in_inference

    # EOS and the output length limit remain independent of user stop sequences.
    req.finish_status = FinishStatus()
    req.update_finish_status([10], output_len=1)
    assert req.finish_status.is_stopped() is True
    req.finish_status = FinishStatus()
    req.stop_sequences = []
    req.sampling_param.shm_param.max_new_tokens = 1
    req.update_finish_status([], output_len=1)
    assert req.finish_status.is_finished_length() is True
