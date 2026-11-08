import hashlib
import json
from copy import deepcopy

import pytest

from test.benchmark.qwen38 import math_quality as math


@pytest.mark.parametrize(
    "value, expected",
    [("1,234.00", "1234"), ("+001.2500", "1.25"), ("-0.00", "0"), ("-42", "-42"), ("0.0010", "0.001")],
)
def test_decimal_normalization_is_exact(value, expected):
    assert math.normalize_number(value) == expected
    large = "123456789012345678901234567890123456789"
    assert math.normalize_number(large) == large


@pytest.mark.parametrize("value", ["1,23", "1.2.3", "1/2", "1e3", "NaN", "inf", "", None])
def test_invalid_plain_numbers_are_not_guessed(value):
    assert math.normalize_number(value) is None


def test_score_uses_final_numeric_answer_only():
    assert math.score_answer("Work: 10 - 4 = 6.\n#### 6.00\n", "6")["correct"]
    assert not math.score_answer("Work: 6 + 1 = 7.\n#### 7", "6")["correct"]
    with pytest.raises(ValueError, match="Expected answer"):
        math.score_answer("#### 6", "unknown")


@pytest.mark.parametrize("text", ["The answer is 6", "#### 6\nMore work", "#### 7\n#### 6", "#### $6", "#### NaN"])
def test_missing_ambiguous_or_nonfinal_answers_are_invalid(text):
    assert math.extract_final_number(text) is None


def test_loader_pins_bytes_and_rejects_duplicate_questions(tmp_path, monkeypatch):
    path = tmp_path / "test.jsonl"
    rows = [
        dict(question="Question one", answer="A calculation.\n#### 4"),
        dict(question="Question two", answer="#### 5"),
    ]
    payload = "\n".join(json.dumps(row) for row in rows).encode()
    path.write_bytes(payload)
    monkeypatch.setattr(math, "DATASET_SHA256", hashlib.sha256(payload).hexdigest())
    examples = math.load_examples(path)
    assert [(row.index, row.expected) for row in examples] == [(0, "4"), (1, "5")]
    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="pinned source"):
        math.load_examples(path)
    duplicate = "\n".join(json.dumps(rows[0]) for _ in range(2)).encode()
    path.write_bytes(duplicate)
    monkeypatch.setattr(math, "DATASET_SHA256", hashlib.sha256(duplicate).hexdigest())
    with pytest.raises(ValueError, match="distinct questions"):
        math.load_examples(path)


def test_sampling_is_reproducible_and_without_replacement():
    examples = tuple(math.MathExample(i, f"Question {i}", "4") for i in range(20))
    selected = math.select_examples(examples, 8, 38)
    assert selected == math.select_examples(examples, 8, 38)
    assert len({row.index for row in selected}) == 8


@pytest.mark.parametrize("limit, seed", [(0, 1), (2, 1), (True, 1), (1, True), (1, 1.0)])
def test_invalid_selection_is_rejected(limit, seed):
    with pytest.raises(ValueError):
        math.select_examples((math.MathExample(0, "Question", "4"),), limit, seed)


def test_prompts_do_not_include_gold_solutions_and_preserve_token_budget():
    class Tokenizer:
        messages = []

        def apply_chat_template(self, messages, **kwargs):
            self.messages.append(messages)
            assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
            return [11, 12]

    tokenizer = Tokenizer()
    first = math.prepare_cases([math.MathExample(0, "What is two plus two?", "4")], tokenizer, 3, 5)
    second = math.prepare_cases([math.MathExample(0, "What is two plus two?", "999")], tokenizer, 3, 5)
    assert tokenizer.messages[0] == tokenizer.messages[1]
    assert first[0]["prompt_ids"] == second[0]["prompt_ids"] == [11, 12]
    assert first[0]["expected_number"] == "4" and second[0]["expected_number"] == "999"
    with pytest.raises(ValueError, match="serving context"):
        math.prepare_cases([math.MathExample(0, "Question", "4")], tokenizer, 4, 5)


@pytest.mark.parametrize("ids", [[], [True], [-1], [1.0], "tokens"])
def test_invalid_tokenizer_payloads_are_rejected(ids):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return ids

    with pytest.raises(ValueError, match="actual nonnegative"):
        math.prepare_cases([math.MathExample(0, "Question", "4")], Tokenizer())


def scored_report():
    return dict(
        status="recorded",
        manifest_sha256="a" * 64,
        scored=2,
        results=[
            dict(
                dataset_index=1, token_ids=[11, 99], finish_reason="stop", truncated=False, correct=True, predicted="4"
            ),
            dict(
                dataset_index=2,
                token_ids=[12, 13],
                finish_reason="length",
                truncated=True,
                correct=False,
                predicted=None,
            ),
        ],
    )


def test_output_comparison_keeps_wrong_and_truncated_cases():
    reference = scored_report()
    assert math.compare_outputs(reference, deepcopy(reference)) == dict(status="passed", cases=2, mismatches=[])
    for field, value in [("token_ids", [12, 14]), ("finish_reason", "stop"), ("truncated", False), ("correct", True)]:
        actual = deepcopy(reference)
        actual["results"][1][field] = value
        assert math.compare_outputs(reference, actual) == dict(status="failed", cases=2, mismatches=[2])


@pytest.mark.parametrize("field,value", [("status", "failed"), ("manifest_sha256", "b" * 64), ("scored", 1)])
def test_output_comparison_rejects_incomplete_or_different_workloads(field, value):
    reference = scored_report()
    actual = deepcopy(reference)
    actual[field] = value
    with pytest.raises(ValueError):
        math.compare_outputs(reference, actual)


def test_output_comparison_rejects_duplicate_or_different_cases():
    reference = scored_report()
    actual = deepcopy(reference)
    for index in (1, 3):
        actual["results"][1]["dataset_index"] = index
        with pytest.raises(ValueError):
            math.compare_outputs(reference, actual)
