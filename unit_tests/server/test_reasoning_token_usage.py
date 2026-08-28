from unittest.mock import patch

from lightllm.server.reasoning_parser import ReasoningParser


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
