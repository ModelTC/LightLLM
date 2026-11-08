from test.benchmark.qwen38.check_generation import REVISION
from test.benchmark.service.stream_cohort import StreamingWorkload


def workloads_from_observations(report, expected_cases):
    if (
        report.get("status") != "observations_recorded"
        or report.get("service", {}).get("framework") not in ("vllm", "sglang")
        or report.get("revision") != REVISION
        or report.get("cases") != expected_cases
        or report.get("reference") is not None
        or report.get("cache_policy") != "cold"
    ):
        raise ValueError("Require explicit cold baseline observations with pinned inputs")
    results, length = report.get("results", {}), report.get("output_length")
    if type(length) is not int or length < 1 or set(results) != set(expected_cases):
        raise ValueError("Require complete baseline observations and a positive output length")
    workloads, differences = [], []
    for name, prompt in expected_cases.items():
        result = results[name]
        ids, body = result["token_ids"], result["response"]
        usage, choices = body["usage"], body["choices"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if (
            len(choices) != 1
            or choices[0]["token_ids"] != ids
            or usage["prompt_tokens"] != len(prompt)
            or usage["completion_tokens"] != length
            or len(ids) != length
            or type(cached) is not int
            or cached != 0
        ):
            raise ValueError("Baseline observations must preserve raw IDs, counts and explicit zero cache")
        work = StreamingWorkload(name, prompt, ids)
        if name.startswith("repeat_"):
            original = name.removeprefix("repeat_")
            if original not in expected_cases or prompt != expected_cases[original]:
                raise ValueError("Repeat observations require identical pinned inputs")
            if ids != results[original]["token_ids"]:
                differences.append(int(original.removeprefix("long_")))
        else:
            workloads.append(work)
    quality = dict(status="failed" if differences else "passed", mismatches=differences)
    if quality != report.get("repeat_consistency"):
        raise ValueError("Baseline repeat-consistency summary differs from actual observations")
    if not workloads:
        raise ValueError("Require a non-repeated workload")
    return workloads, quality
