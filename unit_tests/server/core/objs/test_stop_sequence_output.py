from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.stop_sequence_output import StopSequenceOutput


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
