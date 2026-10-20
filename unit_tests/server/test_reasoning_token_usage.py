from unittest.mock import patch

import pytest

from lightllm.server.reasoning_parser import ReasoningParser
from lightllm.server.httpserver.stop_seq_match import ReasoningStateMachine


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
    def __init__(self, end_ids=(92, 93), start_ids=(90, 91)):
        self.start_ids = list(start_ids)
        self.end_ids = list(end_ids)

    def encode(self, text, **kwargs):
        return {"<think>": self.start_ids, "</think>": self.end_ids}[text]


def test_stop_state_tracks_multi_token_delimiters():
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), 0, [])
    tokens = [1, 90, 91, 2, 92, 93, 3]
    assert [state.handle(token) for token in tokens] == [True, True, False, False, False, False, True]


@pytest.mark.parametrize(
    "tail_prompt_ids,in_reasoning",
    [([90, 91], True), ([1, 90, 91], True), ([90, 91, 1], False), ([90], False), ([], False)],
)
def test_prompt_complete_start_initializes_reasoning(tail_prompt_ids, in_reasoning):
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), 0, tail_prompt_ids)
    assert state.in_reasoning is in_reasoning
    assert state.handle(2) is (not in_reasoning)


@pytest.mark.parametrize("model,force", [("qwen3", 1), ("deepseek-r1", -1)])
def test_prompt_end_marker_preserves_initial_reasoning_hint(model, force):
    state = ReasoningStateMachine(model, _StopTokenizer(), force, [90, 91, 1, 92, 93])
    assert state.state is state.State.REASONING
    assert state.handle(2) is False


@pytest.mark.parametrize(
    "force,tail_prompt_ids,in_reasoning",
    [(0, [90, 91, 1, 2], False), (1, [92, 93, 1, 2], True), (0, [90, 91, 92, 93, 1, 2], False)],
)
def test_prompt_marker_before_trailing_whitespace_does_not_pretransition(force, tail_prompt_ids, in_reasoning):
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), force, tail_prompt_ids)
    assert state.in_reasoning is in_reasoning
    assert state.handle(3) is (not in_reasoning)


@pytest.mark.parametrize("force,marker_ids", [(0, [90, 91]), (1, [92, 93])])
def test_prompt_phase_hint_is_preserved_when_marker_is_outside_tail(force, marker_ids):
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), force, marker_ids + [1] * 20)
    assert state.in_reasoning is bool(force)
    assert state.handle(2) is (not bool(force))


def test_short_prompt_retains_complete_tail_for_long_delimiters():
    tokenizer = _StopTokenizer(end_ids=[94, 95, 96, 97], start_ids=[90, 91, 92])
    state = ReasoningStateMachine("qwen3", tokenizer, 0, [90, 91])
    assert not state.in_reasoning
    assert state.handle(92) is False
    assert state.in_reasoning
    for token in [94, 95, 96, 97]:
        assert state.handle(token) is False
    assert not state.in_reasoning


@pytest.mark.parametrize("model,force", [("qwen3", 1), ("deepseek-r1", -1)])
def test_stop_state_starts_in_forced_reasoning(model, force):
    state = ReasoningStateMachine(model, _StopTokenizer(), force, [])
    assert state.state is state.State.REASONING
    assert state.handle(1) is False
    assert state.handle(92) is False
    assert state.state is state.State.REASONING
    assert state.in_reasoning
    assert state.handle(93) is False
    assert state.handle(2) is True


def test_stop_state_carries_a_partial_delimiter_from_pd_prompt():
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), 1, [1, 92])
    assert state.state is state.State.REASONING
    assert state.in_reasoning
    assert state.handle(93) is False
    assert state.in_reasoning is False
    assert state.handle(2) is True


def test_stop_state_can_reenter_reasoning_after_answer_content():
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), 0, [])
    assert state.state is state.State.CONTENT
    assert state.handle(90) is True
    assert state.state is state.State.CONTENT
    assert state.handle(1) is True  # 尚未完整命中思维链 start 标记，仍处于正文。
    assert state.state is state.State.CONTENT
    for _ in range(2):
        assert state.handle(90) is True
        assert state.handle(91) is False
        assert state.in_reasoning is True
        assert state.handle(1) is False
        assert state.handle(92) is False
        assert state.state is state.State.REASONING
        assert state.in_reasoning
        assert state.handle(93) is False
        assert state.in_reasoning is False
        assert state.handle(2) is True


@pytest.mark.parametrize(
    "token,can_match,next_state",
    [(91, False, "REASONING"), (90, True, "CONTENT"), (1, True, "CONTENT")],
)
def test_stop_state_requires_complete_start_from_prompt(token, can_match, next_state):
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), 0, [1, 90])
    assert state.state is state.State.CONTENT
    assert not state.in_reasoning
    assert state.handle(token) is can_match
    assert state.state is state.State[next_state]


@pytest.mark.parametrize("end_ids,can_match,next_state", [([90, 92], True, "CONTENT"), ([90, 91], False, "REASONING")])
def test_start_matching_only_recognizes_start_marker(end_ids, can_match, next_state):
    state = ReasoningStateMachine("qwen3", _StopTokenizer(end_ids), 0, [])
    assert state.handle(90) is True
    assert state.state is state.State.CONTENT
    assert state.handle(end_ids[-1]) is can_match
    assert state.state is state.State[next_state]


def test_end_marker_outside_reasoning_is_content():
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), 0, [])
    assert state.handle(92) is True
    assert state.handle(93) is True
    assert state.state is state.State.CONTENT


@pytest.mark.parametrize("token,next_state", [(93, "CONTENT"), (92, "REASONING"), (1, "REASONING")])
def test_stop_state_requires_complete_end_from_prompt(token, next_state):
    state = ReasoningStateMachine("qwen3", _StopTokenizer(), 1, [1, 92])
    assert state.state is state.State.REASONING
    assert state.handle(token) is False
    assert state.state is state.State[next_state]
    assert state.in_reasoning is (next_state != "CONTENT")
    assert state.handle(2) is (next_state == "CONTENT")


def test_start_marker_inside_reasoning_is_reasoning_content():
    state = ReasoningStateMachine("qwen3", _StopTokenizer([90, 92]), 1, [])
    assert state.handle(90) is False
    assert state.state is state.State.REASONING
    assert state.handle(91) is False
    assert state.state is state.State.REASONING
    assert state.in_reasoning


@pytest.mark.parametrize(
    "end_ids,tokens,states",
    [
        ([90, 91], [90, 91], ["REASONING", "CONTENT"]),
        ([91], [90, 91], ["REASONING", "CONTENT"]),
        ([91, 92], [90, 91, 92], ["REASONING", "REASONING", "CONTENT"]),
    ],
)
def test_reasoning_matches_end_even_when_it_overlaps_start(end_ids, tokens, states):
    state = ReasoningStateMachine("qwen3", _StopTokenizer(end_ids), 1, [])
    for token, expected_state in zip(tokens, states):
        assert state.handle(token) is False
        assert state.state is state.State[expected_state]
    assert not state.in_reasoning
    assert state.handle(1) is True
