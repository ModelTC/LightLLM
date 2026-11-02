import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import requests


def tokenizer_cache(model, directory, revision):
    snapshot = directory / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        original = model.resolve() / name
        if not original.exists():
            if name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                raise ValueError(f"Missing pinned tokenizer file: {name}")
            continue
        destination = snapshot / name
        if not destination.exists():
            destination.symlink_to(original)
        if destination.read_bytes() != original.read_bytes():
            raise ValueError("Cached tokenizer differs from the pinned checkpoint")
        hashes[name] = hashlib.sha256(original.read_bytes()).hexdigest()
    return hashes


def profiling_records(directory):
    records = [json.loads(line) for line in (directory / "profile_export_raw.jsonl").read_text().splitlines()]
    records = [row for row in records if row["metadata"]["benchmark_phase"] == "profiling"]

    def index(row):
        identity = row["metadata"]["conversation_id"]
        if not identity.startswith("session_") or not identity.removeprefix("session_").isdigit():
            raise ValueError("Unexpected source conversation identity")
        return int(identity.removeprefix("session_"))

    records.sort(key=index)
    if [index(row) for row in records] != list(range(len(records))):
        raise ValueError("Missing or duplicated source conversations")
    return records


def audit_records(directory, trace):
    summary = json.loads((directory / "profile_export_aiperf.json").read_text())
    records = profiling_records(directory)
    if len(records) != len(trace):
        raise ValueError("Incomplete replay")
    if sorted(row["metadata"]["session_num"] for row in records) != list(range(len(trace))):
        raise ValueError("Missing, duplicated or reordered dataset sessions")
    requests_for_comparison, starts, cache_counts, empty_content = [], [], [], []
    for index, (record, expected) in enumerate(zip(records, trace)):
        payload, metadata = record["payload"], record["metadata"]
        if record["status"] != 200 or metadata.get("was_cancelled"):
            raise ValueError("Failed or cancelled request")
        if payload["max_tokens"] != expected["output_length"] or not payload.get("ignore_eos"):
            raise ValueError("Replay changed the specified output work")
        usage, done, finished, has_text = None, False, False, False
        for response in record["responses"]:
            for packet in response.get("packets", []):
                if packet.get("name") != "data":
                    continue
                if packet["value"] == "[DONE]":
                    done = True
                    continue
                data = json.loads(packet["value"])
                usage = data.get("usage") or usage
                finished |= any(choice.get("finish_reason") == "length" for choice in data.get("choices", []))
                has_text |= any(bool(choice.get("text")) for choice in data.get("choices", []))
        if not done or not finished or not usage:
            raise ValueError("Missing stream terminator, length finish or actual usage")
        if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != expected["output_length"]:
            raise ValueError("Actual completion count differs from the trace")
        if (
            type(usage.get("prompt_tokens")) is not int
            or not 0 < usage["prompt_tokens"] + usage["completion_tokens"] <= 4096
        ):
            raise ValueError("Actual request exceeds the shared context limit")
        cache_counts.append((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        if not has_text:
            empty_content.append(index)
        requests_for_comparison.append(
            dict(
                prompt=payload["prompt"],
                max_tokens=payload["max_tokens"],
                temperature=payload["temperature"],
                ignore_eos=payload["ignore_eos"],
            )
        )
        starts.append(metadata["request_start_ns"])

    errors = summary.get("error_summary", [])
    for error in errors:
        detail = error["error_details"]
        if (
            detail["type"] != "InvalidInferenceResultError"
            or "No responses with actual content" not in detail["message"]
        ):
            raise ValueError("AIPerf reported a transport, protocol or other request error")
    if sum(error["count"] for error in errors) != len(empty_content):
        raise ValueError("AIPerf errors do not match the complete empty-text responses")
    if summary["request_count"]["avg"] != len(trace) - len(empty_content):
        raise ValueError("AIPerf successful request count mismatch")
    successful_tokens = sum(row["output_length"] for index, row in enumerate(trace) if index not in empty_content)
    if summary["output_token_count"]["sum"] != successful_tokens:
        raise ValueError("AIPerf successful token count mismatch")

    lag = [(start - starts[0]) / 1e6 - (row["timestamp"] - trace[0]["timestamp"]) for start, row in zip(starts, trace)]
    required = (
        "request_throughput",
        "output_token_throughput",
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
    )
    for name in required:
        if not math.isfinite(summary[name]["avg"]):
            raise ValueError(f"Invalid AIPerf metric: {name}")
    return dict(
        requests=len(records),
        output_tokens=sum(row["output_length"] for row in trace),
        content_errors=len(empty_content),
        error_rate=len(empty_content) / len(trace),
        empty_content_sessions=empty_content,
        successful_output_tokens=successful_tokens,
        metric_scope="AIPerf successful content-bearing responses; empty-text responses remain errors",
        request_manifest_sha256=hashlib.sha256(
            json.dumps(requests_for_comparison, sort_keys=True).encode()
        ).hexdigest(),
        cached_tokens=cache_counts,
        relative_schedule_lag_ms=dict(median=statistics.median(lag), max=max(lag), min=min(lag)),
        metrics={name: summary[name] for name in required},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--aiperf", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--fresh-owned-service",
        action="store_true",
        help="Native replay in a newly owned service with no prior workload requests",
    )
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Run through exp -m "purpose"')
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("Speed must be finite and positive")
    service = json.loads(args.service_metadata.read_text())
    for key in (
        "framework",
        "version",
        "software",
        "command",
        "hardware",
        "configuration",
        "base_url",
        "model_revision",
    ):
        if not service.get(key):
            raise ValueError(f"Missing actual service provenance: {key}")
    if service["status"] not in ("ready", "measuring_aiperf"):
        raise ValueError("AIPerf requires an exclusively owned, ready service")
    manifest = json.loads((args.trace_directory / "manifest.json").read_text())
    raw = (args.trace_directory / "replay.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["trace_sha256"]:
        raise ValueError("Trace checksum mismatch")
    trace = [json.loads(line) for line in raw.splitlines()]
    if len(trace) != manifest["requests"] or manifest["context_length"] != 4096:
        raise ValueError("Trace manifest mismatch")
    run = Path(os.environ["EXP_RUN_DIR"])
    shutil.copyfile(__file__, run / Path(__file__).name)
    version = subprocess.check_output([str(args.aiperf), "--version"], text=True).strip()
    if version != "0.9.0":
        raise ValueError("Comparison is pinned to AIPerf 0.9.0")
    for row in trace:
        row["timestamp"] /= args.speed
    trace_path = run / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    report = dict(
        status="running",
        service=service,
        aiperf_version=version,
        manifest=manifest,
        speed=args.speed,
        concurrency_ceiling=8,
        cache_policy="flush before each replay; prefix reuse within trace enabled",
        precision_scope="Performance only; reuse independent model task-quality results",
        comparison_eligible=False,
    )

    def save():
        (run / "aiperf-replay.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        save()
        cache = args.trace_directory.resolve() / "tokenizer-cache"
        report["tokenizer_sha256"] = tokenizer_cache(args.model_directory, cache, service["model_revision"])
        environment = dict(
            os.environ,
            HF_HUB_CACHE=str(cache),
            HUGGINGFACE_HUB_CACHE=str(cache),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        framework = service["framework"]
        if framework not in ("lightllm", "vllm", "sglang"):
            raise ValueError("Unsupported framework")
        reset = "/reset_prefix_cache" if framework == "vllm" else "/flush_cache"
        if args.fresh_owned_service:
            import psutil

            if framework != "lightllm" or not service["configuration"].get("fresh_aiperf_replay"):
                raise ValueError("Fresh-instance cache isolation is restricted to the native replay owner")
            if not service.get("fresh_service_no_quality_requests") or service.get("checks"):
                raise ValueError("Fresh replay requires a service with no preceding workload requests")
            owner = psutil.Process(service["launcher_pid"])
            allowed = {"qwen38-h100-native-fresh-service.py", "qwen38-h100-aiperf-profile.py"}
            if owner.create_time() != service["launcher_created"] or not any(
                Path(value).name in allowed for value in owner.cmdline()
            ):
                raise ValueError("Fresh replay owner identity mismatch")
            report[
                "cache_policy"
            ] = "New owned native instance per replay; no prior workload requests; prefix reuse within trace enabled"
            report["cache_reset"] = dict(
                method="fresh_owned_service", container=service["container"]["name"], owner_pid=owner.pid
            )
            save()
        else:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(service["base_url"] + reset, timeout=120)
                report["cache_reset"] = dict(status=response.status_code, body=response.text)
                save()
                response.raise_for_status()
                if response.text.strip().lower() == "false":
                    raise ValueError("Server refused cache reset")
                if framework == "vllm" and response.json().get("success") is not True:
                    raise ValueError("vLLM did not confirm prefix-cache reset")
        command = [
            str(args.aiperf),
            "profile",
            "--model",
            str(args.model_directory.resolve()),
            "--tokenizer",
            "Qwen/Qwen3.8-Flash-Next",
            "--tokenizer-revision",
            service["model_revision"],
            "--url",
            service["base_url"],
            "--endpoint-type",
            "completions",
            "--streaming",
            "--input-file",
            str(trace_path),
            "--custom-dataset-type",
            "mooncake_trace",
            "--fixed-schedule",
            "--fixed-schedule-auto-offset",
            "--concurrency",
            "8",
            "--request-count",
            str(len(trace)),
            "--random-seed",
            "42",
            "--use-server-token-count",
            "--export-level",
            "raw",
            "--artifact-dir",
            str(run / "aiperf"),
            "--ui",
            "none",
            "--workers-max",
            "4",
            "--no-auto-plot",
            "--request-timeout-seconds",
            "600",
            "--extra-inputs",
            json.dumps(dict(temperature=0, ignore_eos=True)),
        ]
        report["command"] = command
        save()
        subprocess.run(command, check=True, env=environment)
        report["audit"] = audit_records(run / "aiperf", trace)
        report["status"] = "aiperf_replay_recorded"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
