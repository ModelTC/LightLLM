import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test.benchmark.qwen38 import check_generation


@pytest.fixture
def tokenizer_root(tmp_path, monkeypatch):
    payload = b"pinned tokenizer fixture"
    (tmp_path / "tokenizer.json").write_bytes(payload)
    monkeypatch.setattr(check_generation, "TOKENIZER_SHA256", hashlib.sha256(payload).hexdigest())

    class Tokenizer:
        def encode(self, prompt, **kwargs):
            return SimpleNamespace(ids=[7, 8, 9])

    monkeypatch.setattr(check_generation.Tokenizer, "from_file", lambda path: Tokenizer())
    return tmp_path


def test_default_workload_preserves_prior_long_cases(tokenizer_root):
    cases = check_generation.cases_for(tokenizer_root)
    assert list(cases) == [
        "english",
        "chinese",
        "code_unicode",
        "long_2049",
        "repeat_long_2049",
        "long_4063",
        "repeat_long_4063",
    ]
    assert len(cases["long_2049"]) == 2049
    assert len(cases["long_4063"]) == 4063


def test_custom_lengths_keep_exact_repeated_inputs_without_aliasing(tokenizer_root):
    cases = check_generation.cases_for(tokenizer_root, (16, 31))
    assert len(cases["long_16"]) == 16 and len(cases["long_31"]) == 31
    assert cases["long_16"] == cases["repeat_long_16"]
    assert cases["long_31"] == cases["repeat_long_31"]
    cases["repeat_long_16"][0] = 100
    assert cases["long_16"][0] == 7
    assert "long_2049" not in cases


@pytest.mark.parametrize("lengths", [(), (0,), (-1,), (True,), (2.0,), (16, 16)])
def test_invalid_lengths_fail_before_loading_tokenizer(lengths):
    with pytest.raises(ValueError, match="Long prompt lengths"):
        check_generation.cases_for(Path("missing-tokenizer"), lengths)


def test_custom_lengths_still_require_pinned_tokenizer(tokenizer_root):
    (tokenizer_root / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned checkpoint"):
        check_generation.cases_for(tokenizer_root, (31,))


@pytest.mark.parametrize("framework", ["lightllm", "vllm", "sglang"])
@pytest.mark.parametrize("cached", [0, 1, None, False])
def test_cold_generation_isolates_requests_and_requires_explicit_zero_cache(framework, cached, tmp_path):
    body = (
        dict(prompt_tokens=2, tokens=[dict(id=7, prompt_cache_len=cached), dict(id=8, prompt_cache_len=cached)])
        if framework == "lightllm"
        else dict(
            choices=[dict(token_ids=[7, 8])],
            usage=dict(prompt_tokens=2, prompt_tokens_details=dict(cached_tokens=cached)),
        )
    )
    sent = []
    serialized = json.dumps(body)

    def post(url, json, timeout):
        sent.append(json)
        return SimpleNamespace(text=serialized, json=lambda: body, raise_for_status=lambda: None)

    client = SimpleNamespace(post=post)
    for index in range(2):
        raw = tmp_path / f"response-{index}.json"
        arguments = (client, "http://service", framework, "model", [1, 2], 2, 10)
        if type(cached) is int and cached == 0:
            result = check_generation.generate(*arguments, response_path=raw, cold_cache=True)
            assert result["token_ids"] == [7, 8]
        else:
            with pytest.raises(ValueError, match="zero cached"):
                check_generation.generate(*arguments, response_path=raw, cold_cache=True)
        assert json.loads(raw.read_text()) == body
    if framework == "lightllm":
        assert all(payload["parameters"]["disable_prompt_cache"] is True for payload in sent)
    else:
        assert all(isinstance(payload["cache_salt"], str) and payload["cache_salt"] for payload in sent)
        assert sent[0]["cache_salt"] != sent[1]["cache_salt"]
