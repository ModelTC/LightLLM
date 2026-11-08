import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "test/benchmark/qwen38"))
    return importlib.import_module("check_cpu_cache")


def events():
    result = []
    for rank in range(4):
        common = dict(rank=rank, first_token=1000, request_id=80)
        result.extend(
            [
                dict(common, event="cpu_offload_queued", monotonic=1, pages=10),
                dict(
                    common,
                    event="cpu_extra_load_queued",
                    monotonic=4.5,
                    lexical_layers=[1],
                    slot_fields=["qsa_raw_keys", "qsa_pooled_keys", "qsa_key_cos", "qsa_key_sin"],
                ),
                dict(common, event="cpu_load_queued", monotonic=5, tokens=2368, kv_length=2368, pinned=True),
            ]
        )
    return result


def test_complete_target_cpu_operations_required_on_every_rank(checker):
    proof = checker.validate_cpu_restore(events(), 4, 1000, 0.5, (4, 6))
    assert proof == [dict(rank=rank, offloads=1, restores=1) for rank in range(4)]


@pytest.mark.parametrize("kind", ["cpu_offload_queued", "cpu_load_queued", "cpu_extra_load_queued"])
def test_partial_transfer_evidence_cannot_pass(checker, kind):
    incomplete = [event for event in events() if not (event["rank"] == 3 and event["event"] == kind)]
    with pytest.raises(ValueError, match="TP rank 3"):
        checker.validate_cpu_restore(incomplete, 4, 1000, 0.5, (4, 6))


@pytest.mark.parametrize(
    "kind, field, value",
    [
        ("cpu_load_queued", "pinned", False),
        ("cpu_load_queued", "tokens", 0),
        ("cpu_load_queued", "kv_length", 64),
        ("cpu_load_queued", "monotonic", 3),
        ("cpu_load_queued", "request_id", None),
        ("cpu_extra_load_queued", "request_id", 81),
        ("cpu_extra_load_queued", "slot_fields", ["qsa_raw_keys"]),
        ("cpu_extra_load_queued", "lexical_layers", []),
        ("cpu_offload_queued", "first_token", 1001),
    ],
)
def test_stale_wrong_request_or_incomplete_model_fields_cannot_pass(checker, kind, field, value):
    altered = copy.deepcopy(events())
    for event in altered:
        if event["rank"] == 0 and event["event"] == kind:
            event[field] = value
    with pytest.raises(ValueError, match="TP rank 0"):
        checker.validate_cpu_restore(altered, 4, 1000, 0.5, (4, 6))
