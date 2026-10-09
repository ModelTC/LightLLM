from types import SimpleNamespace

import pytest
import numpy as np

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.core.objs.stop_sequence_output import StopSequenceOutput
from lightllm.server.detokenization import decode_req as decode_req_module
from lightllm.server.detokenization.decode_req import DecodeReq
from lightllm.server.router.model_infer import infer_batch


def _output(stop_strings=("END",), include=False):
    return StopSequenceOutput(
        SimpleNamespace(
            stop_sequences=SimpleNamespace(to_strings=lambda: list(stop_strings)),
            include_stop_str_in_output=include,
        )
    )


def _token(index, text, status=FinishStatus.NO_FINISH):
    return 80, text, {"id": index, "logprob": -index, "count_output_tokens": index}, FinishStatus(status)


@pytest.mark.parametrize("include,expected", [(False, "hello"), (True, "helloEND")])
@pytest.mark.parametrize("chunks", [("helloENDextra",), ("helloE", "", "N", "Dextra")])
def test_stop_output_preserves_token_metadata(include, expected, chunks):
    output = _output(include=include)
    tokens = [
        _token(i, chunk, FinishStatus.FINISHED_STOP if i == len(chunks) else FinishStatus.NO_FINISH)
        for i, chunk in enumerate(chunks, 1)
    ]
    results = [result for token in tokens for result in output.process(*token)]

    assert "".join(result[1] for result in results) == expected
    assert len(results) == len(tokens)
    for result, token in zip(results, tokens):
        assert result[0] == token[0]
        assert result[2] is token[2]
        assert result[3] is token[3]
    assert results[-1][3].get_finish_reason() == "stop"


def test_stream_holds_possible_stop_prefix():
    output = _output()
    assert list(output.process(*_token(1, "hello"))) == []
    results = list(output.process(*_token(2, "E")))
    assert results == []
    assert list(output.process(*_token(3, "N")))[0][1] == "hello"
    results = list(output.process(*_token(4, "D", FinishStatus.FINISHED_STOP)))
    assert [result[1] for result in results] == ["", "", ""]


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize(
    "status",
    [
        FinishStatus.FINISHED_LENGTH,
        FinishStatus.FINISHED_STOP,
        FinishStatus.FINISHED_ABORTED,
        FinishStatus.FINISHED_ERROR,
    ],
)
def test_unmatched_tail_is_flushed(include, status):
    output = _output(include=include)
    results = list(output.process(*_token(1, "helloE")))
    results.extend(output.process(*_token(2, "N", status)))
    assert "".join(result[1] for result in results) == "helloEN"
    assert len(results) == 2


@pytest.mark.parametrize("include,expected", [(False, "hello"), (True, "helloEND")])
def test_multiple_stops_use_first_match(include, expected):
    output = _output(("LATER", "END"), include)
    results = list(output.process(*_token(1, "helloENDLATER", FinishStatus.FINISHED_STOP)))
    assert results[0][1] == expected


@pytest.mark.parametrize("include,expected", [(False, "hello"), (True, "hello\n\n")])
def test_overlapping_stops_prefer_longest_at_same_position(include, expected):
    output = _output(("\n\n", "\n"), include)
    results = list(output.process(*_token(1, "hello\n\nextra", FinishStatus.FINISHED_STOP)))
    assert results[0][1] == expected


@pytest.mark.parametrize("include,expected", [(False, "你好"), (True, "你好结束")])
def test_unicode_stop_split_across_tokens(include, expected):
    output = _output(("结束",), include)
    results = list(output.process(*_token(1, "你好结")))
    results.extend(output.process(*_token(2, "束后缀", FinishStatus.FINISHED_STOP)))
    assert "".join(result[1] for result in results) == expected


def test_no_string_stops_is_immediate():
    output = _output(())
    token = _token(1, "hello")
    assert list(output.process(*token)) == [token]


def test_long_empty_token_run_does_not_lose_stop_prefix():
    output = _output()
    results = list(output.process(*_token(1, "E")))
    for index in range(2, 32):
        results.extend(output.process(*_token(index, "")))
    results.extend(output.process(*_token(32, "ND", FinishStatus.FINISHED_STOP)))
    assert "".join(result[1] for result in results) == ""
    assert len(results) == 32


class _ReasoningTokenizer:
    def encode(self, text, **kwargs):
        return {"<think>": [1000], "</think>": [1001]}.get(text, [ord(c) for c in text])


def _decode_req(monkeypatch, stop_in_reasoning=False, stop="END", force=1, parser="qwen3", pd=False, prompt=(1, 2)):
    monkeypatch.setattr(decode_req_module, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser=parser))
    monkeypatch.setattr(decode_req_module, "get_stop_in_reasoning", lambda: stop_in_reasoning)
    tokenizer = _ReasoningTokenizer()
    params = SamplingParams()
    params.init(tokenizer, stop_sequences=stop, _stop_force_reasoning=force)
    req = SimpleNamespace(
        request_id=80,
        group_req_id=80,
        input_len=len(prompt),
        sample_params=params,
        shm_prompt_ids=SimpleNamespace(arr=np.array(prompt)),
        finish_token_index=-1,
        finish_status=FinishStatus(),
        stop_str_matched=False,
        stop_sequence_match_length=0,
        stop_sequence_match_suffix_length=0,
        _stop_reasoning=False,
    )
    return DecodeReq(req, pd, tokenizer)


@pytest.mark.parametrize("stop_in_reasoning,stop_at", [(False, 3), (True, 0)])
@pytest.mark.parametrize("hidden_delimiter", [False, True])
def test_reasoning_stop_matches_only_selected_scope(monkeypatch, stop_in_reasoning, stop_at, hidden_delimiter):
    decode = _decode_req(monkeypatch, stop_in_reasoning)
    chunks = [(10, "END"), (1001, "" if hidden_delimiter else "</think>"), (11, "answerE"), (12, "NDextra")]
    for index, (token_id, text) in enumerate(chunks):
        decode.output_ids.append(token_id)
        matched = decode.match_stop_sequences(token_id, text)
        assert matched is (index == stop_at)
        if matched:
            assert decode.req.stop_sequence_match_length == 3
            assert decode.req.stop_sequence_match_suffix_length == (5 if index == 3 else 0)
            break


@pytest.mark.parametrize("include", [False, True])
def test_reasoning_eos_preserves_ignored_stop_text(monkeypatch, include):
    decode = _decode_req(monkeypatch)
    output = _output(include=include)
    decode.output_ids.append(10)
    assert decode.match_stop_sequences(10, "reasoning END") is False
    metadata = {"_stop_sequence_match": None, "_stop_reasoning": True}
    results = list(output.process(80, "reasoning END", metadata, FinishStatus(FinishStatus.FINISHED_STOP)))
    assert results[0][1] == "reasoning END"
    assert "_stop_sequence_match" not in results[0][2]
    assert "_stop_reasoning" not in results[0][2]


@pytest.mark.parametrize("include,expected", [(False, "END</think>answer"), (True, "END</think>answerEND")])
def test_scoped_match_filters_content_and_keeps_reasoning(monkeypatch, include, expected):
    decode = _decode_req(monkeypatch)
    output = _output(include=include)
    results = []
    for index, (token_id, text) in enumerate([(10, "END"), (1001, "</think>"), (11, "answerENDextra")]):
        decode.output_ids.append(token_id)
        matched = decode.match_stop_sequences(token_id, text)
        metadata = {}
        if matched:
            metadata["_stop_sequence_match"] = (
                decode.req.stop_sequence_match_length,
                decode.req.stop_sequence_match_suffix_length,
            )
        finish = FinishStatus(FinishStatus.FINISHED_STOP if matched else FinishStatus.NO_FINISH)
        results.extend(output.process(80, text, metadata, finish))
    assert "".join(result[1] for result in results) == expected


def test_stop_cannot_cross_reasoning_content_boundary(monkeypatch):
    decode = _decode_req(monkeypatch)
    for token_id, text in [(1, "E"), (1001, ""), (2, "ND")]:
        decode.output_ids.append(token_id)
        assert decode.match_stop_sequences(token_id, text) is False


def test_token_id_stops_ignore_reasoning_and_match_content(monkeypatch):
    decode = _decode_req(monkeypatch, stop=[[10, 11]])
    for token_id in [10, 11, 1001, 10]:
        decode.output_ids.append(token_id)
        assert decode.match_stop_sequences(token_id, "") is False
    decode.output_ids.append(11)
    assert decode.match_stop_sequences(11, "") is True


def test_disabled_thinking_stops_immediately(monkeypatch):
    decode = _decode_req(monkeypatch, force=0)
    decode.output_ids.append(10)
    assert decode.match_stop_sequences(10, "END") is True


def test_pd_first_token_updates_reasoning_state(monkeypatch):
    decode = _decode_req(monkeypatch, pd=True, prompt=[1, 1001])
    assert decode.match_stop_sequences(1001, "</think>") is False
    decode.output_ids.append(10)
    assert decode.match_stop_sequences(10, "END") is True


def test_pd_capacity_marker_preserves_reasoning_state(monkeypatch):
    decode = _decode_req(monkeypatch)
    decode.output_ids.append(1001)
    decode.req.finish_token_index = decode.input_len
    decode.req.finish_status = FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY)
    assert decode.match_stop_sequences(1001, "") is False
    assert decode.req._stop_reasoning is True


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
