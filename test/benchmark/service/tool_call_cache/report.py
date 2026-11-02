"""Aggregate matched trial pairs without hiding neutral or negative cases.

Bootstrap independent workload seeds, not individual requests. Repeated seeds
across hardware crossover phases form one cluster. Preserve every raw pair.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import random
import statistics


def confidence(values, seed=0):
    center = statistics.mean(values)
    if len(values) < 2:
        return {"mean": center, "ci95": None}
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return {"mean": center, "ci95": [samples[249], samples[9749]]}


def summarize(phases):
    groups = collections.defaultdict(list)
    workloads = {}
    for phase in phases:
        if (phase / "invalidity.json").exists():
            raise ValueError(f"invalidated phase cannot enter a performance report: {phase}")
        progress = json.loads((phase / "progress.json").read_text())
        protocol = json.loads((phase / "protocol.json").read_text())
        trials = json.loads((phase / "summary.json").read_text())
        if progress.get("status") != "complete" or len(trials) != len(protocol["cases"]) * protocol["repeats"]:
            raise ValueError(f"incomplete phase cannot enter final report: {phase}")
        if protocol.get("reject_timed_tilelang_compilation"):
            if not (phase / "jit-audit.json").exists():
                raise ValueError(f"compilation isolation record missing: {phase}")
            audit = json.loads((phase / "jit-audit.json").read_text())
            if audit.get("status") != "complete" or len(audit["trials"]) != len(trials):
                raise ValueError(f"compilation isolation not verified: {phase}")
            if any(set(r["policies"]) != {"candidate", "control"} for r in audit["trials"]):
                raise ValueError(f"compilation audit omits a policy: {phase}")
            if any(
                r["policies"][label][key] != trial[label][key]
                for r, trial in zip(audit["trials"], trials)
                for label in ("candidate", "control")
                for key in ("started_unix", "finished_unix")
            ):
                raise ValueError(f"compilation audit covers different intervals: {phase}")
            if [(r["case"], r["repeat"]) for r in audit["trials"]] != [(r["case"], r["repeat"]) for r in trials] or any(
                p["overlapping_compilations"] for r in audit["trials"] for p in r["policies"].values()
            ):
                raise ValueError(f"timed compilation cannot enter final report: {phase}")
        if protocol.get("gpu_monitor_interval_seconds"):
            observations = [json.loads(line) for line in (phase / "gpu-observations.jsonl").read_text().splitlines()]
            if not observations or any(
                row.get("monitor_error") or row.get("foreign_on_trial_gpus") or row.get("thermal_violations")
                for row in observations
            ):
                raise ValueError(f"GPU isolation not verified: {phase}")
        candidate = json.loads((phase / "candidate-service.json").read_text())
        control = json.loads((phase / "control-service.json").read_text())
        for trial in trials:
            raw = {}
            for policy in ["candidate", "control"]:
                path = phase / f"{trial['case']}-r{trial['repeat']}-{policy}.json"
                rows = json.loads(path.read_text())["rows"]
                if any(
                    row["finish_reason"] not in row.get("allowed_finish_reasons", [row["expected_finish_reason"]])
                    for row in rows
                ):
                    raise ValueError(f"unexpected raw finish state: {path}")
                if any(row["usage"]["prompt_tokens"] != row["client_prompt_tokens"] for row in rows):
                    raise ValueError(f"raw prompt length mismatch: {path}")
                prompt = sum(row["usage"]["prompt_tokens"] for row in rows)
                cached = sum(row["usage"]["prompt_tokens_details"]["cached_tokens"] for row in rows)
                totals = {
                    "requests": len(rows),
                    "prompt_tokens": prompt,
                    "cached_tokens": cached,
                    "prefilled_tokens": prompt - cached,
                }
                if any(trial[policy][key] != value for key, value in totals.items()):
                    raise ValueError(f"summary totals do not match raw rows: {path}")
                raw[policy] = {(row["session"], row["turn"]): row for row in rows}
            if raw["candidate"].keys() != raw["control"].keys():
                raise ValueError(f"raw request keys differ: {phase} {trial['case']}")
            for request, candidate_row in raw["candidate"].items():
                control_row = raw["control"][request]
                for field in ["prompt_sha256", "finish_reason", "output_signature"]:
                    if candidate_row[field] != control_row[field]:
                        raise ValueError(f"raw {field} differs: {phase} {request}")
                if candidate_row["usage"]["completion_tokens"] != control_row["usage"]["completion_tokens"]:
                    raise ValueError(f"generated token counts differ: {phase} {request}")
            if trial["mismatches"]:
                raise ValueError(f"mismatched trial cannot enter matched report: {phase} {trial['case']}")
            if trial["candidate"]["unexpected_end_states"] or trial["control"]["unexpected_end_states"]:
                raise ValueError(f"invalid end states in {phase}")
            config = candidate["configuration"]
            workload = {
                "options": next(case["options"] for case in protocol["cases"] if case["name"] == trial["case"]),
                "harness_sha256": protocol["harness_sha256"],
                "rounds_sha256": protocol.get("rounds_sha256"),
                "control_policy": {key: control["configuration"].get(key) for key in ("retention", "fraction")},
                "cache_isolation": protocol.get("cache_isolation", "none"),
                "image_id": candidate["image_id"],
                "model_config_sha256": candidate["model_config_sha256"],
                "python_dependency_sha256": candidate.get("python_dependency_sha256", {}),
                "service_configuration": {
                    k: config.get(k)
                    for k in ["tp", "max_req_tokens", "batch_tokens", "max_requests", "disable_cudagraph", "log_level"]
                },
                "backend_flags": {
                    flag: candidate["command"][candidate["command"].index(flag) + 1]
                    if flag in candidate["command"]
                    else None
                    for flag in [
                        "--tool_call_parser",
                        "--linear_att_page_block_num",
                        "--llm_prefill_att_backend",
                        "--llm_decode_att_backend",
                        "--prefill_queue_strategy",
                        "--output_constraint_mode",
                    ]
                },
            }
            fingerprint = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
            workloads[fingerprint] = workload
            key = (
                config["model_dir"],
                candidate["source_revision"],
                control["source_revision"],
                config["cache_tokens"],
                config["retention"],
                config["fraction"],
                trial["case"],
                fingerprint,
            )
            groups[key].append({"phase": phase.name, "trial": trial})
    result = []
    for key, pairs in sorted(groups.items()):
        per_seed = collections.defaultdict(list)
        for pair in pairs:
            trial = pair["trial"]
            candidate, control = trial["candidate"], trial["control"]
            metrics = {
                "log_wall_speedup": math.log(control["wall_seconds"] / candidate["wall_seconds"]),
                "log_mean_request_speedup": math.log(
                    control["request_seconds"]["mean"] / candidate["request_seconds"]["mean"]
                ),
                "log_p95_request_speedup": math.log(
                    control["request_seconds"]["p95"] / candidate["request_seconds"]["p95"]
                ),
                "cache_hit_gain_pp": 100 * (candidate["cache_hit_ratio"] - control["cache_hit_ratio"]),
                "prefill_reduction_percent": 100 * (1 - candidate["prefilled_tokens"] / control["prefilled_tokens"]),
            }
            if "server_ttft_ms" in candidate and "server_ttft_ms" in control:
                metrics["log_mean_ttft_speedup"] = math.log(
                    control["server_ttft_ms"]["mean"] / candidate["server_ttft_ms"]["mean"]
                )
                metrics["log_p95_ttft_speedup"] = math.log(
                    control["server_ttft_ms"]["p95"] / candidate["server_ttft_ms"]["p95"]
                )
            if candidate["session_seconds"]["mean"] and control["session_seconds"]["mean"]:
                metrics["log_mean_session_speedup"] = math.log(
                    control["session_seconds"]["mean"] / candidate["session_seconds"]["mean"]
                )
            agent_means = [p.get("agent_session_seconds", {}).get("mean") for p in (candidate, control)]
            if all(agent_means):
                metrics["log_agent_mean_session_speedup"] = math.log(agent_means[1] / agent_means[0])
            for kind, label in (("agent", "agent"), ("one-shot", "one_shot")):
                left = candidate.get("cohort_metrics", {}).get(kind)
                right = control.get("cohort_metrics", {}).get(kind)
                if left is None or right is None:
                    continue
                if right["prefilled_tokens"]:
                    metrics[f"{label}_prefill_reduction_percent"] = 100 * (
                        1 - left["prefilled_tokens"] / right["prefilled_tokens"]
                    )
                for metric_field, suffix in (("request_seconds", "request"), ("server_ttft_ms", "ttft")):
                    if metric_field not in left or metric_field not in right:
                        continue
                    for stat in ("mean", "p95"):
                        if left[metric_field][stat] and right[metric_field][stat]:
                            metrics[f"log_{label}_{stat}_{suffix}_speedup"] = math.log(
                                right[metric_field][stat] / left[metric_field][stat]
                            )
            per_seed[candidate["configuration"]["seed"]].append(metrics)
        clustered = {
            name: [statistics.mean(m[name] for m in rows) for rows in per_seed.values()]
            for name in next(iter(per_seed.values()))[0]
        }
        summary = {
            "model": key[0],
            "candidate_revision": key[1],
            "control_revision": key[2],
            "cache_tokens": key[3],
            "retention_seconds": key[4],
            "fraction": key[5],
            "case": key[6],
            "workload_sha256": key[7],
            "workload": workloads[key[7]],
            "pairs": len(pairs),
            "independent_seeds": len(per_seed),
            "metrics": {},
            "raw_pairs": pairs,
        }
        for name, values in clustered.items():
            estimate = confidence(values)
            if name.startswith("log_"):
                name = name[4:]
                estimate = {
                    "geomean": math.exp(estimate["mean"]),
                    "ci95": ([math.exp(v) for v in estimate["ci95"]] if estimate["ci95"] is not None else None),
                }
            summary["metrics"][name] = estimate
        result.append(summary)
    return result


def public_latency_stats(raw):
    return {
        stat: value
        for stat in ("mean", "p50", "p95")
        if (value := raw.get(stat)) is None or (type(value) in (int, float) and math.isfinite(value))
    }


def public_cohort_metrics(cohorts):
    """Keep only fixed cohort labels and numerical counters/latencies."""
    output = {}
    for kind in ("agent", "one-shot", "trace"):
        raw = cohorts.get(kind)
        if not isinstance(raw, dict):
            continue
        safe = {}
        for key in ("requests", "prompt_tokens", "cached_tokens", "prefilled_tokens", "output_tokens"):
            value = raw.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                safe[key] = value
        for key in ("request_seconds", "server_ttft_ms"):
            if not isinstance(raw.get(key), dict):
                continue
            safe[key] = public_latency_stats(raw[key])
        output[kind] = safe
    return output


def public_summary(result):
    """Export whitelisted metrics; never copy request rows, contents or logs."""
    public = {
        "method": result["method"],
        "limitations": result["limitations"],
        "artifact_scope": "Aggregate and per-trial performance metrics; no request/response contents or raw logs.",
        "groups": [],
        "trials": [],
    }
    metrics = [
        "wall_seconds",
        "started_sessions",
        "completed_sessions",
        "abandoned_sessions",
        "sessions_per_second",
        "session_seconds",
        "requests",
        "prompt_tokens",
        "cached_tokens",
        "cache_hit_ratio",
        "prefilled_tokens",
        "output_tokens",
        "request_seconds",
        "wait_capped_requests",
        "output_target_capped_requests",
        "finish_reason_counts",
        "server_ttft_ms",
    ]
    options = [
        "sessions",
        "turns",
        "split",
        "mix",
        "context_scale",
        "max_prompt_tokens",
        "output_scale",
        "max_output_target",
        "wait_scale",
        "max_wait",
        "start_jitter",
        "compaction_identity",
        "ordinary_output",
        "abandon_fraction",
        "min_tool_wait",
    ]
    for group in result["groups"]:
        safe = {
            key: group[key]
            for key in [
                "candidate_revision",
                "control_revision",
                "cache_tokens",
                "retention_seconds",
                "fraction",
                "case",
                "pairs",
                "independent_seeds",
                "metrics",
                "workload_sha256",
            ]
        }
        safe["model"] = Path(group["model"]).name
        workload = group["workload"]
        safe["workload"] = {
            key: workload[key]
            for key in [
                "harness_sha256",
                "cache_isolation",
                "image_id",
                "model_config_sha256",
                "service_configuration",
                "backend_flags",
            ]
        }
        safe["workload"]["python_dependency_sha256"] = workload.get("python_dependency_sha256", {})
        safe["workload"]["rounds_sha256"] = workload.get("rounds_sha256")
        safe["workload"]["control_policy"] = {
            key: value
            for key in ("retention", "fraction")
            if type(value := workload.get("control_policy", {}).get(key)) in (int, float) and math.isfinite(value)
        }
        safe["workload"]["options"] = {key: workload["options"][key] for key in options if key in workload["options"]}
        public["groups"].append(safe)
        for pair in group["raw_pairs"]:
            trial = pair["trial"]
            entry = {
                "phase": pair["phase"],
                "case": trial["case"],
                "repeat": trial["repeat"],
                "order": trial["order"],
                "mismatch_count": len(trial["mismatches"]),
                "seed": trial["candidate"]["configuration"]["seed"],
            }
            for policy in ["candidate", "control"]:
                entry[policy] = {key: trial[policy][key] for key in metrics if key in trial[policy]}
                for key in ("agent_completed_sessions", "agent_sessions_per_second"):
                    value = trial[policy].get(key)
                    if type(value) in (int, float) and math.isfinite(value):
                        entry[policy][key] = value
                if isinstance(trial[policy].get("agent_session_seconds"), dict):
                    entry[policy]["agent_session_seconds"] = public_latency_stats(
                        trial[policy]["agent_session_seconds"]
                    )
                if "session_kind_counts" in trial[policy]:
                    entry[policy]["session_kind_counts"] = {
                        kind: value
                        for kind in ("agent", "one-shot", "trace")
                        if type(value := trial[policy]["session_kind_counts"].get(kind)) is int and value >= 0
                    }
                if "cohort_metrics" in trial[policy]:
                    entry[policy]["cohort_metrics"] = public_cohort_metrics(trial[policy]["cohort_metrics"])
            public["trials"].append(entry)
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phases", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-output", type=Path, help="Optional whitelisted numerical export, without request contents"
    )
    args = parser.parse_args()
    if "EXP_RUN_DIR" in os.environ:
        shutil.copy2(__file__, Path(os.environ["EXP_RUN_DIR"]) / Path(__file__).name)
    result = {
        "method": "Paired ratios; 95% descriptive bootstrap intervals clustered by workload seed; 10000 resamples.",
        "limitations": "Few seeds give limited uncertainty estimates. Request counts are not independent repetitions.",
        "groups": summarize(args.phases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.public_output:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(json.dumps(public_summary(result), indent=2) + "\n")
    lines = [
        "| Model | Candidate SHA | Control SHA (TTL / budget) | Case | TTL | "
        "Pairs / seeds | Wall speedup | Cache gain (pp) | Prefill reduction |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in result["groups"]:
        metric = group["metrics"]
        speed = metric["wall_speedup"]
        interval = "" if speed["ci95"] is None else f" [{speed['ci95'][0]:.3f}, {speed['ci95'][1]:.3f}]"
        control = group["workload"].get("control_policy", {})
        lines.append(
            f"| {Path(group['model']).name} | {group['candidate_revision'][:8]} | "
            f"{group['control_revision'][:8]} ({control.get('retention', '?')}s / {control.get('fraction', '?')}) | "
            f"{group['case']} | {group['retention_seconds']} | "
            f"{group['pairs']} / {group['independent_seeds']} | {speed['geomean']:.3f}x{interval} | "
            f"{metric['cache_hit_gain_pp']['mean']:+.2f} | {metric['prefill_reduction_percent']['mean']:+.2f}% |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
