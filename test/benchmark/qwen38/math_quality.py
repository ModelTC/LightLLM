from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import random
import re

DATASET_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATASET_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
INSTRUCTION = (
    "Solve the following math problem. Show your reasoning briefly, then end with a final line "
    "exactly in the format #### <number>. Use a plain decimal number.\n\nProblem:\n"
)


def normalize_number(value):
    if not isinstance(value, str) or not NUMBER.fullmatch(value.strip()):
        return None
    number = Decimal(value.strip().replace(",", ""))
    if number == 0:
        return "0"
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def extract_final_number(completion):
    if not isinstance(completion, str):
        return None
    lines = completion.rstrip().splitlines()
    if not lines or sum(line.strip().startswith("####") for line in lines) != 1:
        return None
    match = re.fullmatch(r"####[ \t]+(.+)", lines[-1].strip())
    return normalize_number(match[1]) if match else None


def score_answer(completion, expected):
    gold = normalize_number(expected)
    if gold is None:
        raise ValueError("Expected answer must be a finite plain decimal number")
    predicted = extract_final_number(completion)
    return dict(expected=gold, predicted=predicted, correct=predicted is not None and predicted == gold)


def compare_outputs(reference, actual):
    digest = reference.get("manifest_sha256")
    if (
        reference.get("status") != "recorded"
        or actual.get("status") != "recorded"
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or actual.get("manifest_sha256") != digest
    ):
        raise ValueError("Require complete scored runs with identical workload manifests")

    def rows(report):
        values = report["results"]
        if type(report.get("scored")) is not int or report["scored"] < 1 or len(values) != report["scored"]:
            raise ValueError("Scored count differs from completed cases")
        indices = [row["dataset_index"] for row in values]
        if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("Require distinct actual dataset indices")
        return {row["dataset_index"]: row for row in values}

    expected, observed = rows(reference), rows(actual)
    if set(expected) != set(observed):
        raise ValueError("Scored runs contain different selected cases")
    fields = ("token_ids", "finish_reason", "truncated", "correct", "predicted")
    mismatches = [
        index for index in sorted(expected) if any(expected[index][key] != observed[index][key] for key in fields)
    ]
    return dict(status="passed" if not mismatches else "failed", cases=len(expected), mismatches=mismatches)


@dataclass(frozen=True)
class MathExample:
    index: int
    question: str
    expected: str


def load_examples(path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATASET_SHA256:
        raise ValueError("GSM8K test data differs from the pinned source")
    examples = []
    for index, line in enumerate(payload.splitlines()):
        row = json.loads(line)
        question = row.get("question")
        answer = extract_final_number(row.get("answer"))
        if not isinstance(question, str) or not question.strip() or answer is None:
            raise ValueError("Dataset requires a question and unambiguous final numeric answer")
        examples.append(MathExample(index, question, answer))
    if not examples or len({example.question for example in examples}) != len(examples):
        raise ValueError("Dataset requires distinct questions")
    return tuple(examples)


def select_examples(examples, limit, seed):
    if type(limit) is not int or not 1 <= limit <= len(examples) or type(seed) is not int:
        raise ValueError("Require a valid sample count and integer seed")
    return tuple(examples[index] for index in random.Random(seed).sample(range(len(examples)), limit))


def prepare_cases(examples, tokenizer, max_output_tokens=512, context_limit=4096):
    if any(type(value) is not int or value < 1 for value in (max_output_tokens, context_limit)):
        raise ValueError("Output and context limits must be positive integers")
    cases = []
    for example in examples:
        messages = [{"role": "user", "content": INSTRUCTION + example.question}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Tokenizer must return actual nonnegative integer prompt IDs")
        if len(ids) + max_output_tokens > context_limit:
            raise ValueError("Math example exceeds the declared serving context")
        cases.append(
            dict(
                dataset_index=example.index,
                question_sha256=hashlib.sha256(example.question.encode()).hexdigest(),
                prompt_ids=ids,
                expected_number=example.expected,
                max_output_tokens=max_output_tokens,
            )
        )
    if not cases or len({case["dataset_index"] for case in cases}) != len(cases):
        raise ValueError("Require distinct selected examples")
    return cases
