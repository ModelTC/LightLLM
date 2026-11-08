import hashlib
import json
import re

from test.benchmark.qwen38.check_generation import REVISION

MANIFEST_SHA256 = "3214c9f09a0eb3bae49061d4658daeaea251b4befcfa5214278d796e5762d7cd"
LABELS = ("ALPHA", "BETA", "GAMMA")
ANSWER = re.compile(r"\s*ALPHA\s*=\s*(\d+)\s*;\s*BETA\s*=\s*(\d+)\s*;\s*GAMMA\s*=\s*(\d+)\s*")


def score_records(text, expected):
    if set(expected) != set(LABELS) or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value) for value in expected.values()
    ):
        raise ValueError("Require the three declared record labels and exact digit strings")
    match = ANSWER.fullmatch(text) if isinstance(text, str) else None
    predicted = dict(zip(LABELS, match.groups())) if match else None
    records = {label: predicted is not None and predicted[label] == expected[label] for label in LABELS}
    return dict(predicted=predicted, expected=expected, records=records, correct=all(records.values()))


def load_workload(path, model_directory):
    manifest = json.loads(path.read_text())
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Retrieval workload differs from pinned preparation")
    config = json.loads((model_directory / "config.json").read_text())
    context = config.get("text_config", config)["max_position_embeddings"]
    if (
        manifest["status"] != "prepared"
        or manifest["model_revision"] != REVISION
        or manifest["context_limit"] != context
        or manifest["decoding"] != dict(temperature=0, ignore_eos=False, enable_thinking=False)
    ):
        raise ValueError("Retrieval workload model, context or decoding differs")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if hashlib.sha256((model_directory / name).read_bytes()).hexdigest() != manifest["tokenizer_files"].get(name):
            raise ValueError("Tokenizer differs from retrieval preparation")
    for case in manifest["cases"]:
        ids, limit = case["prompt_ids"], case["max_output_tokens"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or token < 0 for token in ids)
            or type(limit) is not int
            or limit < 1
            or len(ids) + limit > context
        ):
            raise ValueError("Retrieval inputs exceed context or lack exact token IDs")
        score_records("", case["expected_records"])
    return manifest
