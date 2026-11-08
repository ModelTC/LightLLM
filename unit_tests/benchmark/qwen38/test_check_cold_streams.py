from copy import deepcopy

import pytest

from test.benchmark.qwen38.check_cold_streams import workloads_from_reference
from test.benchmark.qwen38.check_generation import REVISION


def reference():
    return dict(
        status="recorded",
        revision=REVISION,
        output_length=2,
        cases=dict(english=[1, 2], long_3=[1, 2, 3], repeat_long_3=[1, 2, 3]),
        results={name: dict(token_ids=[7, 8]) for name in ("english", "long_3", "repeat_long_3")},
    )


def test_qualified_workloads_keep_order_and_exclude_repeat_controls():
    evidence = reference()
    workloads = workloads_from_reference(evidence, evidence["cases"])
    assert [(work.name, work.prompt_ids, work.expected_ids) for work in workloads] == [
        ("english", (1, 2), (7, 8)),
        ("long_3", (1, 2, 3), (7, 8)),
    ]


@pytest.mark.parametrize(
    "field,value", [("status", "failed"), ("revision", "other"), ("output_length", True), ("output_length", 0)]
)
def test_failed_or_unpinned_reference_cannot_authorize_timing(field, value):
    evidence = reference()
    expected = deepcopy(evidence["cases"])
    evidence[field] = value
    with pytest.raises(ValueError):
        workloads_from_reference(evidence, expected)


@pytest.mark.parametrize(
    "change", ["prompt", "missing_case", "extra_case", "truncated", "invalid_id", "repeat_prompt", "repeat_output"]
)
def test_incomplete_or_edited_evidence_cannot_authorize_timing(change):
    evidence = reference()
    expected = deepcopy(evidence["cases"])
    if change == "prompt":
        evidence["cases"]["english"] = [99, 2]
    elif change == "missing_case":
        del evidence["results"]["long_3"]
    elif change == "extra_case":
        evidence["results"]["other"] = dict(token_ids=[7, 8])
    elif change == "truncated":
        evidence["results"]["long_3"]["token_ids"] = [7]
    elif change == "invalid_id":
        evidence["results"]["long_3"]["token_ids"] = [7, True]
    elif change == "repeat_prompt":
        expected["repeat_long_3"] = [2, 2, 3]
        evidence["cases"]["repeat_long_3"] = list(expected["repeat_long_3"])
    elif change == "repeat_output":
        evidence["results"]["repeat_long_3"]["token_ids"] = [7, 9]
    with pytest.raises(ValueError):
        workloads_from_reference(evidence, expected)
