import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil

from test.benchmark.qwen38.check_generation import REVISION, cases_for, verified_checkpoint
from test.benchmark.service.stream_cohort import StreamingWorkload, run_cohort, summarize_cohort
from test.benchmark.service.token_stream import request_stream


def workloads_from_reference(reference, expected_cases):
    if (
        reference.get("status") not in ("passed", "recorded")
        or reference.get("revision") != REVISION
        or reference.get("cases") != expected_cases
    ):
        raise ValueError("Require a qualified generation reference with the exact pinned inputs")
    length = reference.get("output_length")
    if type(length) is not int or length < 1:
        raise ValueError("Reference output length must be a positive integer")
    results = reference.get("results", {})
    if set(results) != set(expected_cases):
        raise ValueError("Reference must contain every declared generation case")
    workloads = []
    for name, prompt in expected_cases.items():
        ids = results[name].get("token_ids")
        if not isinstance(ids, list) or len(ids) != length or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Reference must contain the complete actual output IDs")
        if name.startswith("repeat_"):
            original = name.removeprefix("repeat_")
            if (
                original not in expected_cases
                or prompt != expected_cases[original]
                or ids != results[original]["token_ids"]
            ):
                raise ValueError("Repeated reference prompts and outputs must match")
        else:
            workloads.append(StreamingWorkload(name, prompt, ids))
    if not workloads:
        raise ValueError("Require at least one non-repeated workload")
    return workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Run through exp -m "purpose"')
    if not 1 <= args.repeats <= 20:
        parser.error("Repeats must be between 1 and 20")
    run = Path(os.environ["EXP_RUN_DIR"])
    shutil.copyfile(__file__, run / "cold-stream-cohorts.py")
    verified_checkpoint(args.model_directory)
    service = json.loads(args.service_metadata.read_text())
    if service.get("framework") not in ("lightllm", "vllm", "sglang") or any(
        not service.get(key) for key in ("version", "command", "hardware", "configuration")
    ):
        raise ValueError("Record actual service framework, version, command, hardware and configuration")
    reference = json.loads(args.reference.read_text())
    lengths = [int(name.removeprefix("long_")) for name in reference.get("cases", {}) if name.startswith("long_")]
    workloads = workloads_from_reference(reference, cases_for(args.model_directory, lengths))
    manifest = [dict(name=work.name, prompt_ids=work.prompt_ids, expected_ids=work.expected_ids) for work in workloads]
    report = dict(
        status="running",
        comparison_eligible=False,
        scope=__doc__,
        reason="Shared hardware diagnostic; exact-output and explicit zero-cache requirements remain mandatory",
        service=service,
        reference=str(args.reference),
        revision=REVISION,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        cache_policy="native request cache bypass or independent random baseline cache salt on every request",
        repeats=args.repeats,
        cohorts=[],
    )
    output = run / "cold-cohort.json"

    def save():
        output.write_text(json.dumps(report, indent=2))

    def request(workload, response_path):
        options = (
            dict(disable_prompt_cache=True)
            if service["framework"] == "lightllm"
            else dict(cache_salt=secrets.token_urlsafe(32))
        )
        return request_stream(
            args.base_url,
            service["framework"],
            str(args.model_directory.resolve()),
            list(workload.prompt_ids),
            len(workload.expected_ids),
            600,
            response_path,
            **options,
        )

    try:
        save()
        for repeat in range(args.repeats):
            order = [1, 2, 4, 8]
            order = order[repeat % 4 :] + order[: repeat % 4]
            sequence = workloads[repeat % len(workloads) :] + workloads[: repeat % len(workloads)]
            for concurrency in order:
                result = run_cohort(
                    sequence * 4, concurrency, request, run / f"repeat-{repeat}/concurrency-{concurrency}"
                )
                result["repeat"] = repeat
                transport_complete = all(
                    row.get("result") is not None
                    and (
                        row["status"] == "passed"
                        or row.get("error") == "Output IDs differ from the qualified reference"
                    )
                    for row in result["records"]
                )
                cached = [
                    row.get("result", {}).get("metrics", {}).get("cached_input_tokens") for row in result["records"]
                ]
                cold = all(type(count) is int and count == 0 for count in cached)
                result["zero_cache_observed"] = cold
                if not cold:
                    result.update(
                        status="failed",
                        metrics=None,
                        cache_gate_error="Every request must report exactly zero cached input tokens",
                    )
                result["unqualified_diagnostic_metrics"] = (
                    summarize_cohort([row["result"] for row in result["records"]]) if transport_complete else None
                )
                report["cohorts"].append(result)
                save()
                print(
                    json.dumps(
                        dict(repeat=repeat, concurrency=concurrency, status=result["status"], zero_cache_observed=cold)
                    ),
                    flush=True,
                )
                if result["status"] != "passed":
                    raise ValueError("Cold request quality, timing or cache gate failed; qualified metrics unavailable")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
