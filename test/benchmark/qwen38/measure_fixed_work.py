import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil

from test.benchmark.qwen38.check_cold_streams import workloads_from_reference
from test.benchmark.qwen38.check_generation import cases_for, verified_checkpoint
from test.benchmark.qwen38.generation_observations import workloads_from_observations
from test.benchmark.service.stream_cohort import run_cohort, summarize_cohort
from test.benchmark.service.token_stream import request_stream


def summarize_fixed_work(cohort, workloads):
    expected = {work.name: work for work in workloads}
    results = []
    for record in cohort["records"]:
        if record["status"] != "passed" and record.get("error") != "Output IDs differ from the qualified reference":
            raise ValueError("Incomplete or invalid streaming request")
        work = expected[record["workload"]]
        result = record["result"]
        metrics = result["metrics"]
        if metrics["input_tokens"] != len(work.prompt_ids) or len(result["token_ids"]) != len(work.expected_ids):
            raise ValueError("Actual input/output lengths differ from the workload")
        cached = metrics.get("cached_input_tokens")
        if type(cached) is not int or cached != 0:
            raise ValueError("Every measured request must explicitly report zero cached input tokens")
        results.append(result)
    if len(results) != len(workloads):
        raise ValueError("Missing workload requests")
    return summarize_cohort(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-metadata", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--observed-generation",
        action="store_true",
        help="Measure count-valid baseline observations; retain repeat-quality failures",
    )
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Run through exp -m "purpose"')
    if not 2 <= args.repeats <= 20 or any(c not in (1, 2, 4, 8) for c in args.concurrency):
        parser.error("Require 2–20 repeats and concurrency in 1/2/4/8")
    service = json.loads(args.service_metadata.read_text())
    for key in ("framework", "version", "software", "command", "hardware", "configuration", "base_url"):
        if not service.get(key):
            raise ValueError(f"Missing actual service provenance: {key}")
    if service["framework"] not in ("lightllm", "vllm", "sglang"):
        raise ValueError("Unsupported framework")
    verified_checkpoint(args.model_directory)
    reference = json.loads(args.reference.read_text())
    lengths = [int(name.removeprefix("long_")) for name in reference["cases"] if name.startswith("long_")]
    expected_cases = cases_for(args.model_directory, lengths)
    if args.observed_generation:
        if (
            service["framework"] not in ("vllm", "sglang")
            or reference.get("service", {}).get("framework") != service["framework"]
        ):
            raise ValueError("Observation timing is restricted to matching baseline frameworks")
        workloads, reference_quality = workloads_from_observations(reference, expected_cases)
    else:
        workloads = workloads_from_reference(reference, expected_cases)
        reference_quality = dict(status="qualified_generation")
    run = Path(os.environ["EXP_RUN_DIR"])
    shutil.copyfile(__file__, run / Path(__file__).name)
    manifest = [dict(name=w.name, prompt_ids=w.prompt_ids, output_length=len(w.expected_ids)) for w in workloads]
    report = dict(
        status="running",
        scope=__doc__,
        service=service,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        reference_kind="observations" if args.observed_generation else "qualified_generation",
        reference_quality=reference_quality,
        service_sha256=hashlib.sha256(args.service_metadata.read_bytes()).hexdigest(),
        repeats=args.repeats,
        cohorts=[],
        warmups=[],
        comparison_eligible=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    def save():
        (run / "fixed-work.json").write_text(json.dumps(report, indent=2))

    def request(work, path):
        options = (
            dict(disable_prompt_cache=True)
            if service["framework"] == "lightllm"
            else dict(cache_salt=secrets.token_urlsafe(32))
        )
        return request_stream(
            service["base_url"],
            service["framework"],
            str(args.model_directory.resolve()),
            list(work.prompt_ids),
            len(work.expected_ids),
            600,
            path,
            **options,
        )

    try:
        save()

        for concurrency in args.concurrency:
            cohort = run_cohort(workloads * 4, concurrency, request, run / f"warmup-{concurrency}")
            report["warmups"].append(cohort)
            summarize_fixed_work(cohort, workloads * 4)
            save()
        for repeat in range(args.repeats):
            order = (
                args.concurrency[repeat % len(args.concurrency) :] + args.concurrency[: repeat % len(args.concurrency)]
            )
            sequence = workloads[repeat % len(workloads) :] + workloads[: repeat % len(workloads)]
            for concurrency in order:
                cohort = run_cohort(sequence * 4, concurrency, request, run / f"repeat-{repeat}-c{concurrency}")
                cohort["repeat"] = repeat
                report["cohorts"].append(cohort)
                cohort["fixed_work_metrics"] = summarize_fixed_work(cohort, sequence * 4)
                save()
                print(
                    json.dumps(
                        dict(
                            repeat=repeat,
                            concurrency=concurrency,
                            exact_status=cohort["status"],
                            output_tokens_per_s=cohort["fixed_work_metrics"]["output_tokens_per_s"],
                        )
                    ),
                    flush=True,
                )
        report["status"] = "fixed_work_measured"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
