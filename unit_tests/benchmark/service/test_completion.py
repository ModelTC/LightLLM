from copy import deepcopy

import pytest

from test.benchmark.service.completion import completion_request, completion_tokens


def native(ids=(11, 99), reason="stop"):
    return dict(
        tokens=[dict(id=token) for token in ids],
        prompt_tokens=2,
        count_output_tokens=len(ids),
        finish_reason=reason,
    )


def baseline(ids=(11,), reason="stop"):
    return dict(
        choices=[dict(index=0, token_ids=list(ids), finish_reason=reason)],
        usage=dict(prompt_tokens=2, completion_tokens=len(ids)),
    )


@pytest.mark.parametrize("framework", ["lightllm", "vllm", "sglang"])
def test_request_sends_only_prompt_and_decoding_options(framework):
    endpoint, payload = completion_request(framework, "model", [1, 2], 3)
    if framework == "lightllm":
        assert endpoint == "/generate" and payload["inputs"] == [1, 2]
        assert payload["parameters"]["print_eos_token"]
        assert payload["parameters"]["ignore_eos"] is False
    else:
        assert endpoint == "/v1/completions" and payload["prompt"] == [1, 2]
        assert payload["ignore_eos"] is False and payload["return_token_ids"]
    assert "expected_number" not in payload


@pytest.mark.parametrize("framework, factory", [("lightllm", native), ("vllm", baseline), ("sglang", baseline)])
def test_natural_eos_and_budget_exhaustion_remain_distinct(framework, factory):
    result = completion_tokens(factory(), framework, [1, 2], 3, [99])
    assert result["finish_reason"] == "stop" and not result["truncated"]
    truncated = completion_tokens(factory((11, 12, 13), "length"), framework, [1, 2], 3, [99])
    assert truncated["token_ids"] == [11, 12, 13] and truncated["truncated"]


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_baseline_may_include_or_omit_terminal_eos(framework):
    for ids in [(11,), (11, 99)]:
        assert completion_tokens(baseline(ids), framework, [1, 2], 3, [99])["token_ids"] == list(ids)


@pytest.mark.parametrize("framework, factory", [("lightllm", native), ("vllm", baseline), ("sglang", baseline)])
@pytest.mark.parametrize("ids, reason", [((99, 11), "stop"), ((11,), "length"), ((11, 12, 99), "length")])
def test_invalid_termination_cannot_be_scored(framework, factory, ids, reason):
    with pytest.raises(ValueError):
        completion_tokens(factory(ids, reason), framework, [1, 2], 3, [99])


def test_native_stop_requires_observed_eos():
    with pytest.raises(ValueError, match="declared EOS"):
        completion_tokens(native((11,)), "lightllm", [1, 2], 3, [99])


@pytest.mark.parametrize("ids", [[], [True], [-1], [1.0], [11, 12, 13, 99]])
def test_invalid_output_ids_and_over_budget_are_rejected(ids):
    with pytest.raises(ValueError):
        completion_tokens(baseline(ids), "vllm", [1, 2], 3, [99])


def test_usage_and_prompt_identity_are_checked():
    original = baseline()
    for field, value in [("prompt_tokens", 3), ("completion_tokens", 2), ("completion_tokens", True)]:
        body = deepcopy(original)
        body["usage"][field] = value
        with pytest.raises(ValueError, match="counts"):
            completion_tokens(body, "vllm", [1, 2], 3, [99])
    body = deepcopy(original)
    body["choices"][0]["prompt_token_ids"] = [1, 3]
    with pytest.raises(ValueError, match="exact prompt"):
        completion_tokens(body, "vllm", [1, 2], 3, [99])
    body["choices"] *= 2
    with pytest.raises(ValueError, match="one completion"):
        completion_tokens(body, "vllm", [1, 2], 3, [99])


@pytest.mark.parametrize("prompt, limit", [([], 3), ([True], 3), ([-1], 3), ([1], True), ([1], 0)])
def test_bad_requests_fail_before_transport(prompt, limit):
    with pytest.raises(ValueError):
        completion_request("lightllm", "model", prompt, limit)
