import importlib.util
import copy
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/report.py"
spec = importlib.util.spec_from_file_location("tool_call_report", path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.mark.parametrize("failure", ["missing", "incomplete", "policy", "clock", "pair", "compilation"])
def test_report_rejects_unproven_compilation_isolation(tmp_path, failure):
    trial = {
        "case": "trace",
        "repeat": 0,
        "candidate": {"started_unix": 10, "finished_unix": 20},
        "control": {"started_unix": 30, "finished_unix": 40},
    }
    policies = {label: {**trial[label], "overlapping_compilations": []} for label in ("candidate", "control")}
    audit = {"status": "complete", "trials": [{"case": "trace", "repeat": 0, "policies": copy.deepcopy(policies)}]}
    for name, data in {
        "progress.json": {"status": "complete"},
        "protocol.json": {"cases": [{"name": "trace"}], "repeats": 1, "reject_timed_tilelang_compilation": True},
        "summary.json": [trial],
    }.items():
        (tmp_path / name).write_text(json.dumps(data))
    if failure == "incomplete":
        audit["status"] = "in_progress"
    elif failure == "policy":
        audit["trials"][0]["policies"].pop("control")
    elif failure == "clock":
        audit["trials"][0]["policies"]["candidate"]["started_unix"] = 11
    elif failure == "pair":
        audit["trials"][0]["repeat"] = 1
    elif failure == "compilation":
        audit["trials"][0]["policies"]["candidate"]["overlapping_compilations"] = [{"started_unix": 12}]
    if failure != "missing":
        (tmp_path / "jit-audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="compilation"):
        report.summarize([tmp_path])


def test_quarantined_hardware_phase_cannot_enter_performance_report(tmp_path):
    (tmp_path / "invalidity.json").write_text(json.dumps({"reason": "thermal slowdown"}))
    (tmp_path / "progress.json").write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="invalidated phase"):
        report.summarize([tmp_path])


def test_public_export_omits_payloads_and_private_configuration():
    secret = "NEVER_PUBLISH_REQUEST_CONTENT"
    policy = {"configuration": {"seed": 42, "url": secret}, "wall_seconds": 1.0, "messages": secret}
    group = dict.fromkeys(
        [
            "candidate_revision",
            "control_revision",
            "cache_tokens",
            "retention_seconds",
            "fraction",
            "case",
            "pairs",
            "independent_seeds",
            "workload_sha256",
        ],
        1,
    )
    group.update(
        model="/private/model/path/Qwen3-4B",
        metrics={"wall_speedup": {"geomean": 1.0}},
        workload={
            "harness_sha256": "a" * 64,
            "cache_isolation": "capacity flood",
            "image_id": "sha256:test",
            "model_config_sha256": "b" * 64,
            "service_configuration": {"tp": 1},
            "backend_flags": {},
            "options": {"sessions": 4, "namespace": secret},
        },
        raw_pairs=[
            {
                "phase": "validation",
                "trial": {
                    "case": "trace",
                    "repeat": 0,
                    "order": ["control", "candidate"],
                    "mismatches": [],
                    "candidate": policy,
                    "control": policy,
                    "raw_response": secret,
                },
            }
        ],
    )
    result = report.public_summary({"method": "paired", "limitations": "few seeds", "groups": [group]})
    assert secret not in json.dumps(result)
    assert "/private/model/path" not in json.dumps(result)
    assert result["groups"][0]["model"] == "Qwen3-4B"
    assert result["groups"][0]["workload"]["options"] == {"sessions": 4}
    assert result["trials"][0]["candidate"] == {"wall_seconds": 1.0}


def test_report_rejects_equal_parsed_outputs_with_different_generated_token_counts(tmp_path):
    def save(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    save("progress.json", {"status": "complete"})
    save("protocol.json", {"cases": [{"name": "case", "options": {}}], "repeats": 1})
    save("candidate-service.json", {})
    save("control-service.json", {})
    totals = {"requests": 1, "prompt_tokens": 1, "cached_tokens": 0, "prefilled_tokens": 1}
    save("summary.json", [{"case": "case", "repeat": 0, "candidate": totals, "control": totals, "mismatches": []}])
    for policy, tokens in [("candidate", 2), ("control", 3)]:
        row = {
            "session": "s",
            "turn": 0,
            "finish_reason": "stop",
            "expected_finish_reason": "stop",
            "client_prompt_tokens": 1,
            "prompt_sha256": "same",
            "output_signature": {"content": "OK", "calls": []},
            "usage": {"prompt_tokens": 1, "completion_tokens": tokens, "prompt_tokens_details": {"cached_tokens": 0}},
        }
        save(f"case-r0-{policy}.json", {"rows": [row]})
    with pytest.raises(ValueError, match="generated token counts differ"):
        report.summarize([tmp_path])
