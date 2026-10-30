"""Run sequential, alternating, exp-recorded paired trials on existing services."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
import threading


def thermal_events(load_lines, gpu_ids, baseline):
    """Catch thermal bursts between samples using cumulative event counters."""
    events = []
    for line in load_lines:
        fields = [part.strip() for part in line.split(",")]
        index = fields[0]
        if index not in gpu_ids:
            continue
        counters = tuple(int(value.split()[0]) for value in fields[9:11])
        if len(counters) != 2:
            raise ValueError("thermal counters are unavailable")
        initial = baseline.setdefault(index, counters)
        if any(current < previous for current, previous in zip(counters, initial)):
            raise ValueError("thermal event counters reset during the experiment")
        deltas = [max(0, current - previous) for current, previous in zip(counters, initial)]
        if any(value.lower() == "active" for value in fields[7:9]) or any(deltas):
            events.append({"gpu": index, "temperature": fields[4], "sm_clock": fields[5], "counter_deltas_us": deltas})
    return events


def monitor_gpus(services, destination):
    """Retain host load observations and reject shared experiment GPUs."""
    gpu_ids = {gpu.strip() for service in services.values() for gpu in service["configuration"]["gpu"].split(",")}
    uuid_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    uuids = {uuid.strip() for line in uuid_lines for index, uuid in [line.split(",")] if index.strip() in gpu_ids}
    if len(uuids) != len(gpu_ids):
        raise ValueError("could not identify every experiment GPU")
    allowed_pids = set()
    for service in services.values():
        rows = subprocess.check_output(
            ["docker", "top", service["configuration"]["name"], "-eo", "pid"], text=True
        ).splitlines()
        allowed_pids.update(int(row.strip()) for row in rows[1:] if row.strip())
    errors = []
    thermal_baseline = {}
    stop = threading.Event()

    def poll():
        with (destination / "gpu-observations.jsonl").open("x") as output:
            final_sample = False
            while True:
                try:
                    apps = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader"],
                        text=True,
                        timeout=10,
                    ).splitlines()
                    foreign = []
                    for row in apps:
                        uuid, pid, _ = [part.strip() for part in row.split(",")]
                        if uuid in uuids and int(pid) not in allowed_pids:
                            foreign.append({"gpu_uuid": uuid, "pid": int(pid)})
                    load = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,utilization.gpu,power.draw,memory.used,temperature.gpu,"
                            "clocks.current.sm,clocks.current.memory,clocks_event_reasons.sw_thermal_slowdown,"
                            "clocks_event_reasons.hw_thermal_slowdown,"
                            "clocks_event_reasons_counters.sw_thermal_slowdown,"
                            "clocks_event_reasons_counters.hw_thermal_slowdown",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=10,
                    ).splitlines()
                    thermal = thermal_events(load, gpu_ids, thermal_baseline)
                    record = {
                        "time_unix": time.time(),
                        "apps": apps,
                        "load": load,
                        "host_load_average": os.getloadavg(),
                        "foreign_on_trial_gpus": foreign,
                        "thermal_violations": thermal,
                    }
                    if foreign or thermal:
                        errors.append(record)
                except Exception as exc:
                    record = {"time_unix": time.time(), "monitor_error": str(exc)}
                    errors.append(record)
                output.write(json.dumps(record) + "\n")
                output.flush()
                if final_sample:
                    break
                # Always sample once after the parent finishes the workload;
                # a thermal burst at the last request must not escape detection.
                final_sample = stop.wait(2)

    worker = threading.Thread(target=poll, daemon=True)
    worker.start()
    return errors, stop, worker


def manifest(name):
    candidates = []
    for path in (Path.home() / "experiments/runs").glob("*/service-manifest.json"):
        data = json.loads(path.read_text())
        if data["configuration"]["name"] == name:
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        raise ValueError(f"no exp service manifest for {name}")
    return max(candidates, key=lambda item: item[0])[1:]


def timings(name, output):
    text = subprocess.check_output(["docker", "logs", name], stderr=subprocess.STDOUT, text=True)
    output.write_text(text)
    pattern = re.compile(r"X-Request-Id:(\S+) .*?first_token_cost:([0-9.eE+-]+)ms .*?gpu_prompt_cache_len:(\d+)")
    return {m[1]: {"ttft_ms": float(m[2]), "gpu_cached_tokens": int(m[3])} for m in pattern.finditer(text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reverse-order", action="store_true", help="Reverse alternating order for GPU crossover")
    parser.add_argument("--model-dir", default="/models/Qwen3-4B")
    parser.add_argument("--rounds", default="cachewise-analysis-v2.rounds.jsonl")
    parser.add_argument("--exp", default="/mtc/sufubao/shared_home/sufubao/.local/bin/exp")
    parser.add_argument(
        "--allow-different-source", action="store_true", help="Explicitly permit the original-base audit"
    )
    parser.add_argument("--no-reset", action="store_true", help="Warmup/exploration only: retain prior cache contents")
    parser.add_argument("--reject-timed-tilelang-compilation", action="store_true")
    args = parser.parse_args()
    destination = args.root / "results" / args.phase
    destination.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    services = {}
    for label, name in [("candidate", args.candidate), ("control", args.control)]:
        source, data = manifest(name)
        state = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
        if not state["State"]["Running"]:
            raise RuntimeError(f"service {name} is not running")
        services[label] = data
        shutil.copy2(source, destination / f"{label}-service.json")
    left, right = services["candidate"], services["control"]
    for key in ["image_id", "model_config_sha256", "python_dependency_sha256"]:
        if left.get(key) != right.get(key):
            raise ValueError(f"services differ in {key}")
    for key in [
        "model_dir",
        "model_name",
        "tp",
        "cache_tokens",
        "max_req_tokens",
        "batch_tokens",
        "max_requests",
        "disable_cudagraph",
        "log_level",
    ]:
        if left["configuration"].get(key) != right["configuration"].get(key):
            raise ValueError(f"services differ in {key}")
    for flag in [
        "--tool_call_parser",
        "--linear_att_page_block_num",
        "--llm_prefill_att_backend",
        "--llm_decode_att_backend",
        "--prefill_queue_strategy",
        "--output_constraint_mode",
    ]:
        values = [
            service["command"][service["command"].index(flag) + 1] if flag in service["command"] else None
            for service in [left, right]
        ]
        if values[0] != values[1]:
            raise ValueError(f"services differ in {flag}")
    if left["source_revision"] != right["source_revision"] and not args.allow_different_source:
        raise ValueError("policy comparison requires identical source; use --allow-different-source for the base audit")
    harness_sha = hashlib.sha256((args.root / "replay.py").read_bytes()).hexdigest()
    shutil.copy2(__file__, destination / "run_paired.py")
    shutil.copy2(args.root / "replay.py", destination / "replay.py")
    if not args.no_reset:
        shutil.copy2(args.root / "reset_cache.py", destination / "reset_cache.py")
    (destination / "protocol.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "repeats": args.repeats,
                "harness_sha256": harness_sha,
                "rounds_sha256": hashlib.sha256((args.root / args.rounds).read_bytes()).hexdigest(),
                "paired_runner_sha256": hashlib.sha256((destination / "run_paired.py").read_bytes()).hexdigest(),
                "gpu_monitor_interval_seconds": 2,
                "reject_timed_tilelang_compilation": args.reject_timed_tilelang_compilation,
                "compilation_guard_sha256": (
                    hashlib.sha256((args.root / "compile_guard.py").read_bytes()).hexdigest()
                    if args.reject_timed_tilelang_compilation
                    else None
                ),
                "thermal_monitoring": "active flags and cumulative counter deltas on experiment GPUs",
                "cache_isolation": "none" if args.no_reset else "expire preferences, then independent capacity flood",
                "reset_sha256": (
                    None if args.no_reset else hashlib.sha256((destination / "reset_cache.py").read_bytes()).hexdigest()
                ),
                "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            indent=2,
        )
        + "\n"
    )
    summaries = []
    compilation_audits = []
    if args.reject_timed_tilelang_compilation:
        from compile_guard import audit_trial, contaminated

        shutil.copy2(args.root / "compile_guard.py", destination / "compile_guard.py")
    gpu_errors, gpu_stop, gpu_worker = monitor_gpus(services, destination)
    last_done = {label: time.monotonic() for label in services}
    for case in cases:
        for repeat in range(args.repeats):
            namespace = f"{args.phase}-{case['name']}-r{repeat}"
            outputs = {}
            order = ["control", "candidate"] if repeat % 2 == 0 else ["candidate", "control"]
            if args.reverse_order:
                order.reverse()
            for label in order:
                if gpu_errors:
                    raise RuntimeError(f"GPU isolation could not be verified; retain observations in {destination}")
                service = services[label]
                output = destination / f"{case['name']}-r{repeat}-{label}.json"
                if not args.no_reset:
                    config = service["configuration"]
                    wait = max(0, config["retention"] + 0.25 - (time.monotonic() - last_done[label]))
                    reset = [
                        args.exp,
                        "-m",
                        f"Untimed isolated-cache prelude {namespace} {label}; "
                        f"service {service['source_revision']}; full manifest in {destination}",
                        "python3",
                        str(destination / "reset_cache.py"),
                        "--url",
                        f"http://127.0.0.1:{config['port']}",
                        "--model",
                        config["model_name"],
                        "--cache-tokens",
                        str(config["cache_tokens"]),
                        "--max-req-tokens",
                        str(config["max_req_tokens"]),
                        "--sleep-seconds",
                        str(wait),
                        "--seed",
                        str(int(hashlib.sha256(namespace.encode()).hexdigest(), 16)),
                    ]
                    print(f"Resetting cache before {namespace} {label}, lease wait {wait:.2f}s", flush=True)
                    with output.with_suffix(".reset.log").open("w") as log:
                        subprocess.run(reset, stdout=log, stderr=subprocess.STDOUT, check=True)
                command = [
                    args.exp,
                    "-m",
                    f"{namespace} {label}; service {service['source_revision']}; "
                    f"image {service['image_id']}; harness {harness_sha}; full manifests in {destination}",
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "host",
                    "--entrypoint",
                    "python",
                    "-e",
                    "HF_HOME=/tmp/hf",
                    "-v",
                    f"{args.root}:/bench",
                    "-v",
                    "/mtc/models:/models:ro",
                    service["image_id"],
                    "/bench/" + str((destination / "replay.py").relative_to(args.root)),
                    "--url",
                    f"http://127.0.0.1:{service['configuration']['port']}",
                    "--rounds",
                    f"/bench/{args.rounds}",
                    "--model-dir",
                    args.model_dir,
                    "--model",
                    service["configuration"]["model_name"],
                    "--namespace",
                    namespace,
                    "--seed",
                    str(42 + repeat),
                    "--output",
                    "/bench/" + str(output.relative_to(args.root)),
                ]
                for key, value in case["options"].items():
                    command.extend(["--" + key.replace("_", "-"), str(value)])
                (destination / "progress.json").write_text(
                    json.dumps(
                        {
                            "case": case["name"],
                            "repeat": repeat,
                            "policy": label,
                            "command": command,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"Starting {namespace} {label}", flush=True)
                with output.with_suffix(".log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                last_done[label] = time.monotonic()
                if gpu_errors:
                    raise RuntimeError(
                        f"GPU interference or monitoring failure; retain trial and observations: {output}"
                    )
                if completed.returncode:
                    raise RuntimeError(f"trial failed, preserve logs: {output.with_suffix('.log')}")
                outputs[label] = json.loads(output.read_text())
            left = {(r["session"], r["turn"]): r for r in outputs["control"]["rows"]}
            right = {(r["session"], r["turn"]): r for r in outputs["candidate"]["rows"]}
            if left.keys() != right.keys():
                raise RuntimeError(f"paired requests differ: {namespace}")
            mismatches = []
            for key in left:
                for field in ["prompt_sha256", "finish_reason", "output_signature"]:
                    if left[key][field] != right[key][field]:
                        mismatches.append({"request": key, "field": field})
                if left[key]["usage"]["completion_tokens"] != right[key]["usage"]["completion_tokens"]:
                    mismatches.append({"request": key, "field": "completion_tokens"})
            summary = {"case": case["name"], "repeat": repeat, "order": order, "mismatches": mismatches}
            for label, result in outputs.items():
                summary[label] = {k: v for k, v in result.items() if k not in ["rows", "sessions"]}
            summaries.append(summary)
            (destination / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            if args.reject_timed_tilelang_compilation:
                logs = {
                    label: subprocess.check_output(
                        ["docker", "logs", services[label]["configuration"]["name"]],
                        text=True,
                        stderr=subprocess.STDOUT,
                    )
                    for label in ("candidate", "control")
                }
                audit = audit_trial(summary, logs)
                compilation_audits.append(audit)
                (destination / "jit-audit.json").write_text(
                    json.dumps({"status": "in_progress", "trials": compilation_audits}, indent=2) + "\n"
                )
                if contaminated(audit):
                    (destination / "invalidity.json").write_text(
                        json.dumps({"reason": "logged TileLang compilation overlaps measured replay", "trial": audit})
                        + "\n"
                    )
                    raise RuntimeError("Timed compilation detected; preserve phase and warm kernels before retrying")
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "repeat": repeat,
                        "mismatches": len(mismatches),
                        **{
                            label: {
                                "wall_seconds": outputs[label]["wall_seconds"],
                                "cache_hit_ratio": outputs[label]["cache_hit_ratio"],
                                "prefilled_tokens": outputs[label]["prefilled_tokens"],
                            }
                            for label in outputs
                        },
                    }
                ),
                flush=True,
            )
            if mismatches:
                raise RuntimeError(
                    f"paired inputs/outputs differ; retain but do not interpret as matched performance: {namespace}"
                )
    for label, name in [("candidate", args.candidate), ("control", args.control)]:
        measured = timings(name, destination / f"{label}-server.log")
        for summary in summaries:
            data = json.loads((destination / f"{summary['case']}-r{summary['repeat']}-{label}.json").read_text())
            values = [measured[r["request_id"]]["ttft_ms"] for r in data["rows"]]
            if any(
                measured[r["request_id"]]["gpu_cached_tokens"] != r["usage"]["prompt_tokens_details"]["cached_tokens"]
                for r in data["rows"]
            ):
                raise RuntimeError("server/client cache metrics differ")
            summary[label]["server_ttft_ms"] = {
                "mean": statistics.mean(values),
                "p95": sorted(values)[int((len(values) - 1) * 0.95)],
            }
            cohorts = summary[label].get("cohort_metrics", {})
            for kind in cohorts:
                subset = [
                    measured[r["request_id"]]["ttft_ms"] for r in data["rows"] if r.get("request_kind", "trace") == kind
                ]
                cohorts[kind]["server_ttft_ms"] = {
                    "mean": statistics.mean(subset),
                    "p95": sorted(subset)[int((len(subset) - 1) * 0.95)],
                }
    gpu_stop.set()
    gpu_worker.join(timeout=25)
    if gpu_worker.is_alive() or gpu_errors:
        raise RuntimeError(f"GPU isolation could not be verified; retain observations in {destination}")
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if args.reject_timed_tilelang_compilation:
        (destination / "jit-audit.json").write_text(
            json.dumps({"status": "complete", "trials": compilation_audits}, indent=2) + "\n"
        )
    (destination / "progress.json").write_text(
        json.dumps({"status": "complete", "paired_trials": len(summaries)}) + "\n"
    )
    print(f"Completed {len(summaries)} paired trials: {destination}", flush=True)


if __name__ == "__main__":
    main()
