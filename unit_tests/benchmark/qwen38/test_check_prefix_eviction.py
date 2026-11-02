import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "test/benchmark/qwen38"))
    return importlib.import_module("check_prefix_eviction")


def events():
    result = []
    for rank in range(4):
        result.extend(
            [
                dict(event="state_freed", rank=rank, monotonic=2.0, pool=10, indexes=[3]),
                dict(event="node_evicted", rank=rank, monotonic=2.1, first_token=1000, big_pool=10, big_state=3),
                dict(event="state_freed", rank=rank, monotonic=3.0, pool=20, indexes=[4]),
                dict(
                    event="small_state_evicted",
                    rank=rank,
                    monotonic=3.1,
                    first_token=1000,
                    small_pool=20,
                    small_state=4,
                ),
            ]
        )
    return result


def test_actual_target_big_and_small_state_releases_required_on_every_rank(checker):
    proof = checker.validate_eviction(events(), 4, 1000, 1, 4)
    assert {item["rank"] for item in proof} == {0, 1, 2, 3}


@pytest.mark.parametrize("missing", ["node_evicted", "small_state_evicted", "state_freed"])
def test_partial_eviction_evidence_cannot_pass(checker, missing):
    incomplete = [event for event in events() if event["event"] != missing]
    with pytest.raises(ValueError, match="No observed target"):
        checker.validate_eviction(incomplete, 4, 1000, 1, 4)


def test_one_rank_trace_cannot_prove_four_rank_eviction(checker):
    with pytest.raises(ValueError, match="rank 1"):
        checker.validate_eviction([event for event in events() if event["rank"] == 0], 4, 1000, 1, 4)


@pytest.mark.parametrize("change", ["other_target", "other_pool", "other_index", "late_free", "outside_interval"])
def test_unrelated_or_out_of_order_events_cannot_prove_target_eviction(checker, change):
    changed = copy.deepcopy(events())
    for event in changed:
        if change == "other_target" and "first_token" in event:
            event["first_token"] = 1001
        elif change == "other_pool" and event["event"] == "state_freed":
            event["pool"] += 50
        elif change == "other_index" and event["event"] == "state_freed":
            event["indexes"] = [900]
        elif change == "late_free" and event["event"] == "state_freed":
            event["monotonic"] = 3.9
        elif change == "outside_interval":
            event["monotonic"] += 10
    with pytest.raises(ValueError, match="No observed target"):
        checker.validate_eviction(changed, 4, 1000, 1, 4)


@pytest.mark.parametrize("value", [-1, True, 2305, 4.0])
def test_invalid_cache_hit_accounting_rejected(checker, value):
    result = dict(response=dict(prompt_tokens=2305, tokens=[dict(prompt_cache_len=value)]))
    with pytest.raises(ValueError, match="cache hit"):
        checker.cache_hit(result)


def test_pressure_prefixes_diverge_before_first_hash_page(checker, monkeypatch):
    monkeypatch.setattr(checker, "cases_for", lambda _: {"long_4063": [9] * 4063})
    cases = checker.eviction_cases(Path("unused"))
    assert len({tuple(case["inputs"][:64]) for case in cases.values()}) == 33
    assert (len(cases["target"]["inputs"]) - 1) % 256 == 64
    assert all(
        len(case["inputs"]) == (65 if name.startswith("small_pressure_") else 2369)
        and len(case["inputs"]) + case["max_new_tokens"] <= 4096
        for name, case in cases.items()
    )
