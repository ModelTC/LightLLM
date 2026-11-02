import hashlib
import json

import pytest

from test.benchmark.qwen38 import retrieval_quality as quality

EXPECTED = dict(ALPHA="00123", BETA="456", GAMMA="789")


def test_exact_digit_strings_and_all_records_are_required():
    assert quality.score_records(" ALPHA = 00123;\n BETA=456; GAMMA=789\n", EXPECTED)["correct"]
    result = quality.score_records("ALPHA=123; BETA=456; GAMMA=789", EXPECTED)
    assert not result["correct"] and result["records"] == dict(ALPHA=False, BETA=True, GAMMA=True)


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "ALPHA=00123; BETA=456",
        "ALPHA=00123; BETA=456; GAMMA=789.\n",
        "ALPHA=00123; BETA=456; GAMMA=789; ALPHA=123",
        "Here: ALPHA=00123; BETA=456; GAMMA=789",
    ],
)
def test_incomplete_or_ambiguous_answers_cannot_pass(text):
    result = quality.score_records(text, EXPECTED)
    assert result["predicted"] is None and not result["correct"]


def test_manifest_and_actual_tokenizer_are_pinned(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(dict(text_config=dict(max_position_embeddings=10))))
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        payload = name.encode()
        (tmp_path / name).write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    manifest = dict(
        status="prepared",
        model_revision=quality.REVISION,
        context_limit=10,
        decoding=dict(temperature=0, ignore_eos=False, enable_thinking=False),
        tokenizer_files=files,
        cases=[dict(prompt_ids=[1, 2], max_output_tokens=8, expected_records=EXPECTED)],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        quality, "MANIFEST_SHA256", hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    )
    assert quality.load_workload(path, tmp_path) == manifest
    manifest["cases"][0]["expected_records"]["ALPHA"] = "999"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pinned preparation"):
        quality.load_workload(path, tmp_path)
    manifest["cases"][0]["expected_records"]["ALPHA"] = "00123"
    path.write_text(json.dumps(manifest))
    (tmp_path / "chat_template.jinja").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Tokenizer"):
        quality.load_workload(path, tmp_path)
