import importlib.util
import copy
import json
import shutil
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


def test_cohort_export_whitelists_nested_numeric_fields_and_fixed_labels():
    secret = "NEVER_PUBLISH_COHORT_CONTENT"
    raw = {
        "agent": {
            "requests": 4,
            "prompt_tokens": 100,
            "messages": secret,
            "request_seconds": {"mean": 1.0, "p95": 2.0, "url": secret, "p50": secret},
            "server_ttft_ms": {"mean": float("inf"), "p95": 30, "private": secret},
        },
        secret: {"requests": 9},
        "one-shot": {"requests": 8, "prompt_tokens": secret},
    }
    result = report.public_cohort_metrics(raw)
    assert secret not in json.dumps(result)
    assert result["agent"]["requests"] == 4
    assert result["one-shot"] == {"requests": 8}
    assert result["agent"]["request_seconds"] == {"mean": 1.0, "p95": 2.0}
    assert result["agent"]["server_ttft_ms"]["p95"] == 30
    assert "mean" not in result["agent"]["server_ttft_ms"]


@pytest.mark.parametrize("separation", ["identical", "control_policy", "dataset"])
def test_cohort_aggregation_preserves_group_identity_and_exposes_ordinary_regression(tmp_path, separation):
    def save(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    save("progress.json", {"status": "complete"})
    save(
        "protocol.json",
        {"cases": [{"name": "mix", "options": {"mix": "agent50"}}], "repeats": 1, "harness_sha256": "f" * 64},
    )
    policies = {}
    for policy, revision, cached, agent_latency, ordinary_latency in [
        ("candidate", "a" * 40, 5, 1.0, 4.0),
        ("control", "b" * 40, 0, 2.0, 3.0),
    ]:
        config = {"model_dir": "/model", "cache_tokens": 100, "retention": 3, "fraction": 0.5}
        save(
            f"{policy}-service.json",
            {
                "configuration": config,
                "source_revision": revision,
                "image_id": "image",
                "model_config_sha256": "c" * 64,
                "command": [],
            },
        )
        rows = []
        for index, (kind, prompt, hit) in enumerate([("agent", 10, cached), ("one-shot", 20, 0)]):
            reason = "tool_calls" if kind == "agent" else "stop"
            rows.append(
                {
                    "session": str(index),
                    "turn": 0,
                    "request_kind": kind,
                    "finish_reason": reason,
                    "expected_finish_reason": reason,
                    "client_prompt_tokens": prompt,
                    "prompt_sha256": str(index),
                    "output_signature": {"content": "same"},
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": hit},
                    },
                }
            )
        save(f"mix-r0-{policy}.json", {"rows": rows})
        policies[policy] = {
            "configuration": {"seed": 42},
            "requests": 2,
            "prompt_tokens": 30,
            "cached_tokens": cached,
            "prefilled_tokens": 30 - cached,
            "unexpected_end_states": [],
            "wall_seconds": 10,
            "cache_hit_ratio": cached / 30,
            "request_seconds": {"mean": 2, "p95": 4},
            "session_seconds": {"mean": 8},
            "agent_session_seconds": {"mean": agent_latency * 5},
            "cohort_metrics": {
                "agent": {
                    "prefilled_tokens": 10 - cached,
                    "request_seconds": {"mean": agent_latency, "p95": agent_latency},
                },
                "one-shot": {
                    "prefilled_tokens": 20,
                    "request_seconds": {"mean": ordinary_latency, "p95": ordinary_latency},
                },
            },
        }
    save(
        "summary.json", [{"case": "mix", "repeat": 0, "order": ["control", "candidate"], "mismatches": [], **policies}]
    )
    result = report.summarize([tmp_path])
    assert len(result) == 1
    group = result[0]
    assert group["model"] == "/model" and group["candidate_revision"] == "a" * 40
    assert group["case"] == "mix" and group["workload"]["options"]["mix"] == "agent50"
    assert group["metrics"]["agent_mean_request_speedup"]["geomean"] == pytest.approx(2.0)
    assert group["metrics"]["one_shot_mean_request_speedup"]["geomean"] == pytest.approx(0.75)

    alternate = tmp_path / "alternate"
    alternate.mkdir()
    for source in tmp_path.glob("*.json"):
        shutil.copy2(source, alternate / source.name)
    if separation == "control_policy":
        manifest_path = alternate / "control-service.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["configuration"]["fraction"] = 0.25
        manifest_path.write_text(json.dumps(manifest))
    elif separation == "dataset":
        protocol_path = alternate / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["rounds_sha256"] = "d" * 64
        protocol_path.write_text(json.dumps(protocol))
    combined = report.summarize([tmp_path, alternate])
    assert len(combined) == (1 if separation == "identical" else 2)
    assert sum(group["pairs"] for group in combined) == 2
