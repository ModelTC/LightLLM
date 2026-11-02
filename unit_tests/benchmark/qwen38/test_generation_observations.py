import copy

import pytest

from test.benchmark.qwen38.check_cold_streams import workloads_from_reference
from test.benchmark.qwen38.check_generation import REVISION
from test.benchmark.qwen38.generation_observations import workloads_from_observations


def observations():
    cases = {"long_16": [1, 2], "repeat_long_16": [1, 2]}
    results = {}
    for name, ids in [("long_16", [7, 8]), ("repeat_long_16", [7, 9])]:
        results[name] = dict(
            token_ids=ids,
            response=dict(
                choices=[dict(token_ids=ids)],
                usage=dict(prompt_tokens=2, completion_tokens=2, prompt_tokens_details=dict(cached_tokens=0)),
            ),
        )
    return dict(
        status="observations_recorded",
        service=dict(framework="vllm"),
        revision=REVISION,
        cases=cases,
        reference=None,
        cache_policy="cold",
        results=results,
        output_length=2,
        repeat_consistency=dict(status="failed", mismatches=[16]),
    )


def test_observations_retain_repeat_failure_and_cannot_qualify_a_reference():
    report = observations()
    before = copy.deepcopy(report)
    workloads, quality = workloads_from_observations(report, report["cases"])
    assert report == before
    assert len(workloads) == 1 and workloads[0].expected_ids == (7, 8)
    assert quality == dict(status="failed", mismatches=[16])
    with pytest.raises(ValueError, match="qualified"):
        workloads_from_reference(report, report["cases"])


@pytest.mark.parametrize("change", ["native", "cached", "count", "edited_ids", "false_pass", "incomplete"])
def test_observations_reject_invalid_protocol_or_hidden_quality_failure(change):
    report = observations()
    result = report["results"]["long_16"]
    if change == "native":
        report["service"]["framework"] = "lightllm"
    elif change == "cached":
        result["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
    elif change == "count":
        result["response"]["usage"]["completion_tokens"] = 1
    elif change == "edited_ids":
        result["response"]["choices"][0]["token_ids"] = [7, 10]
    elif change == "false_pass":
        report["repeat_consistency"] = dict(status="passed", mismatches=[])
    elif change == "incomplete":
        del report["results"]["repeat_long_16"]
    with pytest.raises(ValueError):
        workloads_from_observations(report, report["cases"])
