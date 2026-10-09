from unittest.mock import patch

import pytest

from lightllm.server.reasoning_parser import ReasoningParser, ReasoningStopState


def _create_parser(model_type: str, force_reasoning: bool) -> ReasoningParser:
    with patch("lightllm.server.reasoning_parser.get_token_id", return_value=99):
        return ReasoningParser(model_type, force_reasoning=force_reasoning)


def _count_tokens(parser: ReasoningParser, token_ids: list[int]) -> int:
    for token_id in token_ids:
        parser.update_reasoning_token_count(token_id)
    return parser.reasoning_tokens


def test_counts_tokens_until_single_token_closing_marker():
    parser = _create_parser("qwen3", force_reasoning=True)

    reasoning_tokens = _count_tokens(parser, [1, 2, 3, 99, 4])

    assert reasoning_tokens == 3


def test_counts_all_tokens_when_generation_is_truncated_before_closing_marker():
    parser = _create_parser("qwen3", force_reasoning=True)

    reasoning_tokens = _count_tokens(parser, [1, 2])

    assert reasoning_tokens == 2


def test_does_not_count_when_reasoning_is_disabled():
    parser = _create_parser("qwen3", force_reasoning=False)

    reasoning_tokens = _count_tokens(parser, [1, 2, 99])

    assert reasoning_tokens == 0


def test_counts_for_always_reasoning_detector():
    parser = _create_parser("deepseek-r1", force_reasoning=False)

    reasoning_tokens = _count_tokens(parser, [1, 2, 99])

    assert reasoning_tokens == 2


def test_minimax_append_think_output_is_not_counted_as_reasoning():
    parser = _create_parser("minimax-append-think", force_reasoning=True)

    reasoning_tokens = _count_tokens(parser, [1, 2, 99])

    assert reasoning_tokens == 0


class _StopTokenizer:
    def encode(self, text, **kwargs):
        return {"<think>": [90, 91], "</think>": [92, 93]}[text]


def test_stop_state_tracks_multi_token_delimiters():
    state = ReasoningStopState("qwen3", _StopTokenizer(), 0, [])
    tokens = [1, 90, 91, 2, 92, 93, 3]
    assert [state.update(token) for token in tokens] == [True, False, False, False, False, False, True]


@pytest.mark.parametrize("model,force", [("qwen3", 1), ("deepseek-r1", -1)])
def test_stop_state_starts_in_forced_reasoning(model, force):
    state = ReasoningStopState(model, _StopTokenizer(), force, [])
    assert state.update(1) is False
    assert state.update(92) is False
    assert state.update(93) is False
    assert state.update(2) is True


def test_stop_state_carries_a_partial_delimiter_from_pd_prompt():
    state = ReasoningStopState("qwen3", _StopTokenizer(), 1, [1, 92])
    assert state.update(93) is False
    assert state.in_reasoning is False
    assert state.update(2) is True
