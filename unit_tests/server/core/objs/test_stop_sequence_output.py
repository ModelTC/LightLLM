from types import SimpleNamespace

import pytest
import numpy as np

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.core.objs.out_token_circlequeue import CircularQueue
from lightllm.server.detokenization import stop_sequence as stop_sequence_module
from lightllm.server.detokenization.decode_req import DecodeReq
from lightllm.server.detokenization.manager import DeTokenizationManager
from lightllm.server.detokenization.stop_sequence import StopSequenceBuffer
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
    )
    return params


@pytest.fixture(autouse=True)
def _stop_environment(monkeypatch):
    monkeypatch.setattr(
        stop_sequence_module,
        "get_env_start_args",
        lambda: SimpleNamespace(reasoning_parser=None),
    )
    monkeypatch.setattr(stop_sequence_module, "get_stop_in_reasoning", lambda: False)


def _decode_req(
    monkeypatch,
    stop_in_reasoning=False,
    stop="END",
    force=1,
    parser="qwen3",
    include=False,
    tokens=(),
    finish=FinishStatus.NO_FINISH,
):
    monkeypatch.setattr(
        stop_sequence_module,
        "get_env_start_args",
        lambda: SimpleNamespace(reasoning_parser=parser),
    )
    monkeypatch.setattr(stop_sequence_module, "get_stop_in_reasoning", lambda: stop_in_reasoning)
    req = SimpleNamespace(
        request_id=80,
        group_req_id=80,
        input_len=2,
        sample_params=_params(stop, include, force),
        shm_prompt_ids=SimpleNamespace(arr=np.array([1, 2, *tokens])),
        finish_token_index=1 + len(tokens) if tokens else -1,
        finish_status=FinishStatus(finish),
        stop_str_matched=False,
        stop_str_matched_token_index=-1,
        candetoken_out_len=len(tokens),
        out_tokens_queue=CircularQueue(),
        can_released_mark=False,
    )
    return DecodeReq(req, _ReasoningTokenizer())


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


@pytest.mark.parametrize("stop_in_reasoning,stop_at", [(False, 3), (True, 0)])
@pytest.mark.parametrize("hidden_delimiter", [False, True])
def test_reasoning_stop_matches_only_selected_scope(monkeypatch, stop_in_reasoning, stop_at, hidden_delimiter):
    decode = _decode_req(monkeypatch, stop_in_reasoning)
    chunks = [(10, "END"), (1001, "" if hidden_delimiter else "</think>"), (11, "answerE"), (12, "NDextra")]
    results = []
    for index, (token_id, text) in enumerate(chunks):
        decode.output_ids.append(token_id)
        matched = decode.match_stop_sequences(token_id, text)
        results.extend(decode.stop_buffer.ready_tokens)
        decode.stop_buffer.ready_tokens.clear()
        assert matched is (index == stop_at)
        if matched:
            break
    assert "".join(text for text, _ in results) == (
        "END" + ("" if hidden_delimiter else "</think>") + "answer" if stop_at == 3 else ""
    )


@pytest.mark.parametrize("include", [False, True])
def test_reasoning_eos_preserves_ignored_stop_text(monkeypatch, include):
    decode = _decode_req(monkeypatch, include=include)
    decode.output_ids.append(10)
    assert not decode.match_stop_sequences(10, "reasoning END")
    assert list(decode.stop_buffer.ready_tokens) == [("reasoning END", (2, 1))]


@pytest.mark.parametrize("include", [False, True])
def test_scoped_match_filters_content_and_keeps_reasoning(monkeypatch, include):
    decode = _decode_req(monkeypatch, include=include)
    for token_id, text in [(10, "END"), (1001, "</think>"), (11, "answerENDextra")]:
        decode.output_ids.append(token_id)
        decode.match_stop_sequences(token_id, text)
    assert "".join(text for text, _ in decode.stop_buffer.ready_tokens) == (
        "END</think>answer" + ("END" if include else "")
    )


def test_stop_cannot_cross_reasoning_content_boundary(monkeypatch):
    decode = _decode_req(monkeypatch)
    for token_id, text in [(1, "E"), (1001, ""), (2, "ND")]:
        decode.output_ids.append(token_id)
        assert not decode.match_stop_sequences(token_id, text)


def test_token_id_stops_ignore_reasoning_and_match_content(monkeypatch):
    decode = _decode_req(monkeypatch, stop=[[10, 11]])
    for token_id in [10, 11, 1001, 10]:
        decode.output_ids.append(token_id)
        assert not decode.match_stop_sequences(token_id, "")
    decode.output_ids.append(11)
    assert decode.match_stop_sequences(11, "")


def test_pd_first_token_updates_reasoning_state(monkeypatch):
    decode = _decode_req(monkeypatch)
    assert decode.read_offset == decode.input_len
    decode.output_ids.append(1001)
    assert not decode.match_stop_sequences(1001, "</think>")
    decode.output_ids.append(10)
    assert decode.match_stop_sequences(10, "END")


def test_pd_capacity_marker_preserves_reasoning_state(monkeypatch):
    decode = _decode_req(monkeypatch)
    decode.output_ids.append(1001)
    decode.req.finish_token_index = decode.input_len
    decode.req.finish_status = FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY)
    assert not decode.match_stop_sequences(1001, "")
    assert decode.stop_buffer.reasoning_state.in_reasoning is True


@pytest.mark.parametrize(
    "finish",
    [
        FinishStatus.FINISHED_LENGTH,
        FinishStatus.FINISHED_STOP,
        FinishStatus.FINISHED_ERROR,
        FinishStatus.FINISHED_ABORTED,
    ],
)
@pytest.mark.parametrize("matched_stop", [False, True])
@pytest.mark.parametrize("forward_raw", [False, True])
def test_detokenization_drains_buffer_before_release(monkeypatch, finish, matched_stop, forward_raw):
    tokens = list(range(10, 30))
    decode = _decode_req(monkeypatch, force=0, parser=None, tokens=tokens, finish=finish)
    chunks = {tokens[0]: "helloE", tokens[-2]: "NDextra" if matched_stop else "X"}
    monkeypatch.setattr(
        "lightllm.server.detokenization.manager.decode_token",
        lambda tokenizer, req, token_id, eos: chunks.get(token_id, ""),
    )
    manager = object.__new__(DeTokenizationManager)
    manager.req_id_to_out = {80: decode}
    manager.tokenizer = _ReasoningTokenizer()
    manager.eos_id = []
    manager.all_special_ids = set()
    manager.forward_raw_text = forward_raw
    manager.pub_to_httpserver = SimpleNamespace(send_pyobj=lambda *a, **kw: None)
    manager.shm_req_manager = SimpleNamespace(put_back_req_obj=lambda req: None)
    results = []
    for _ in range(100):
        manager.gen_token_out()
        if decode.out_queue_is_full() and not forward_raw:
            assert not decode.req.can_released_mark
        while not decode.req.out_tokens_queue.is_empty():
            results.append(decode.req.out_tokens_queue.pop())
        if decode.req.can_released_mark:
            break
    assert decode.req.can_released_mark
    assert "".join(text for text, *_ in results) == (
        "helloENDextra" if matched_stop and forward_raw else "hello" if matched_stop else "helloEX"
    )
    expected_count = len(tokens) - int(matched_stop)
    assert [count for *_, count in results] == list(range(1, expected_count + 1))
    assert decode.req.stop_str_matched is matched_stop


@pytest.mark.parametrize(
    "parser,enabled,should_stop", [(None, False, True), ("qwen3", False, False), ("qwen3", True, True)]
)
def test_inference_stop_gate_preserves_legacy_and_opt_in(monkeypatch, parser, enabled, should_stop):
    monkeypatch.setattr(infer_batch, "get_stop_in_reasoning", lambda: enabled)
    req = infer_batch.InferReq.__new__(infer_batch.InferReq)
    req.args = SimpleNamespace(reasoning_parser=parser)
    req.stop_sequences = [[10]]
    req.shm_req = SimpleNamespace(input_len=1, shm_prompt_ids=SimpleNamespace(arr=np.array([1, 10])))
    req.sampling_param = SimpleNamespace(shm_param=SimpleNamespace(ignore_eos=False, max_new_tokens=100))
    req.finish_status = FinishStatus()
    req.update_finish_status([], output_len=1)
    assert req.finish_status.is_stopped() is should_stop

    # EOS and the output length limit remain independent of user stop sequences.
    req.finish_status = FinishStatus()
    req.update_finish_status([10], output_len=1)
    assert req.finish_status.is_stopped() is True
    req.finish_status = FinishStatus()
    req.stop_sequences = []
    req.sampling_param.shm_param.max_new_tokens = 1
    req.update_finish_status([], output_len=1)
    assert req.finish_status.is_finished_length() is True
