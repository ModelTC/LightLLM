from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from test.benchmark.qwen38 import check_math_quality as check
from test.benchmark.qwen38.math_quality import MathExample


@pytest.fixture
def workload(tmp_path, monkeypatch):
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        payload = name.encode()
        (tmp_path / name).write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(check, "TOKENIZER_SHA256", files["tokenizer.json"])
    examples = (MathExample(0, "What is two plus two?", "4"),)
    monkeypatch.setattr(check, "load_examples", lambda path: examples)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

    tokenizer = Tokenizer()
    manifest = dict(
        status="prepared",
        model_revision=check.REVISION,
        dataset=dict(revision=check.DATASET_REVISION, sha256=check.DATASET_SHA256, split="test", records=1),
        tokenizer_files=files,
        decoding=dict(temperature=0, ignore_eos=False, enable_thinking=False),
        sample_count=1,
        sample_seed=38,
        max_output_tokens=3,
        context_limit=5,
        cases=check.prepare_cases(examples, tokenizer, 3, 5),
    )
    return manifest, tmp_path, tokenizer


def test_verifies_reconstructed_prompts_and_gold_labels(workload):
    manifest, root, tokenizer = workload
    assert check.verified_workload(manifest, root, root / "data", tokenizer) == manifest["cases"]
    for field, changed in [("expected_number", "999"), ("prompt_ids", [1, 3]), ("dataset_index", 1)]:
        forged = deepcopy(manifest)
        forged["cases"][0][field] = changed
        with pytest.raises(ValueError, match="prompts or gold"):
            check.verified_workload(forged, root, root / "data", tokenizer)


def test_rejects_changed_tokenizer_and_wrong_protocol(workload):
    manifest, root, tokenizer = workload
    forged = deepcopy(manifest)
    forged["decoding"]["ignore_eos"] = True
    with pytest.raises(ValueError, match="greedy non-thinking"):
        check.verified_workload(forged, root, root / "data", tokenizer)
    (root / "chat_template.jinja").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Tokenizer files"):
        check.verified_workload(manifest, root, root / "data", tokenizer)


@pytest.mark.parametrize("bad_count", [False, True])
def test_client_keeps_wrong_and_truncated_answers_in_denominator(tmp_path, monkeypatch, bad_count):
    metadata = dict(
        framework="lightllm", version="fixture", command=["fixture"], hardware="fixture", configuration={"tp": 1}
    )
    (tmp_path / "service.json").write_text(json.dumps(metadata))
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "generation_config.json").write_text(json.dumps(dict(eos_token_id=[99])))
    cases = [dict(dataset_index=i, prompt_ids=[1, 2], max_output_tokens=3, expected_number="4") for i in range(3)]

    class Tokenizer:
        eos_token_id = 99

        def decode(self, ids, **kwargs):
            return {11: "#### 4", 12: "#### 5", 13: "#### 4"}[ids[0]]

    class Session:
        index = 0

        def post(self, url, *, json, timeout):
            assert "expected_number" not in json and json["inputs"] == [1, 2]
            ids, reason = [([11, 99], "stop"), ([12, 99], "stop"), ([13, 14, 15], "length")][self.index]
            self.index += 1
            body = dict(
                tokens=[dict(id=token) for token in ids],
                prompt_tokens=2,
                count_output_tokens=1 if bad_count else len(ids),
                finish_reason=reason,
            )
            return SimpleNamespace(text="fixture", raise_for_status=lambda: None, json=lambda: body)

        def close(self):
            pass

    monkeypatch.setattr(check, "get_tokenizer", lambda path: SimpleNamespace(tokenizer=Tokenizer()))
    monkeypatch.setattr(check, "verified_checkpoint", lambda path: None)
    monkeypatch.setattr(check, "verified_workload", lambda *args: cases)
    monkeypatch.setattr(check.requests, "Session", Session)
    monkeypatch.setenv("EXP_ACTIVE", "1")
    monkeypatch.setenv("EXP_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_math_quality.py",
            "--framework",
            "lightllm",
            "--base-url",
            "http://fixture",
            "--model-directory",
            str(tmp_path),
            "--service-metadata",
            str(tmp_path / "service.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--dataset",
            str(tmp_path / "data"),
        ],
    )
    if bad_count:
        with pytest.raises(ValueError, match="counts"):
            check.main()
        report = json.loads((tmp_path / "math-quality.json").read_text())
        assert report["status"] == "failed" and "accuracy" not in report
    else:
        check.main()
        report = json.loads((tmp_path / "math-quality.json").read_text())
        assert report["status"] == "recorded"
        assert (report["scored"], report["correct"], report["truncated"], report["accuracy"]) == (3, 1, 1, 1 / 3)
