"""Export numerical workload characteristics without request/response contents."""

import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0}
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("workload measurements must be finite numbers")

    def percentile(fraction):
        position = fraction * (len(values) - 1)
        lower = int(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "n": len(values),
        "min": values[0],
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": values[-1],
    }


def summarize(phases):
    result = {
        "scope": "Descriptive per-request workload distributions, not independent performance repetitions. "
        "Percentiles use linear interpolation. All fields are numerical or fixed experiment labels; "
        "request IDs, text, signatures, raw logs and private configuration are omitted.",
        "trials": [],
    }
    fields = [
        "source_input_tokens",
        "source_output_tokens",
        "target_prompt_tokens",
        "target_output_tokens",
        "source_wait_seconds",
        "scaled_wait_seconds",
        "injected_wait_seconds",
        "wait_seconds",
    ]
    for phase in phases:
        if (phase / "invalidity.json").exists():
            raise ValueError(f"invalidated phase: {phase}")
        if json.loads((phase / "progress.json").read_text()).get("status") != "complete":
            raise ValueError(f"incomplete phase: {phase}")
        protocol = json.loads((phase / "protocol.json").read_text())
        trials = json.loads((phase / "summary.json").read_text())
        if len(trials) != len(protocol["cases"]) * protocol["repeats"]:
            raise ValueError(f"missing trial records: {phase}")
        for trial in trials:
            policies = {}
            for policy in ["candidate", "control"]:
                data = json.loads((phase / f"{trial['case']}-r{trial['repeat']}-{policy}.json").read_text())
                rows = data["rows"]
                config = data["configuration"]
                reasons = collections.Counter(row["finish_reason"] for row in rows)
                kinds = collections.Counter(row.get("request_kind", "trace") for row in rows)
                if not set(kinds).issubset({"trace", "agent", "one-shot"}):
                    raise ValueError("unexpected request kind")
                if not set(reasons).issubset({"stop", "length", "tool_calls"}):
                    raise ValueError(f"unexpected finish state: {phase}")
                if any(row["usage"]["prompt_tokens"] != row["client_prompt_tokens"] for row in rows):
                    raise ValueError(f"prompt token mismatch: {phase}")
                policies[policy] = {
                    "requests": len(rows),
                    "request_kind_counts": dict(kinds),
                    "compacted_requests": sum(bool(row["compaction"]) for row in rows),
                    "wait_capped_requests": sum(bool(row["wait_capped"]) for row in rows),
                    "output_target_capped_requests": data["output_target_capped_requests"],
                    "prompt_target_capped_requests": sum(
                        round(row["source_input_tokens"] * config["context_scale"]) > config["max_prompt_tokens"]
                        for row in rows
                    ),
                    "finish_reason_counts": dict(reasons),
                    "prompt_tokens": distribution(row["usage"]["prompt_tokens"] for row in rows),
                    "completion_tokens": distribution(row["usage"]["completion_tokens"] for row in rows),
                    **{field: distribution(row.get(field) for row in rows) for field in fields},
                }
            if policies["candidate"] != policies["control"]:
                raise ValueError(f"paired workload characteristics differ: {phase} {trial['case']}")
            result["trials"].append(
                {"phase": phase.name, "case": trial["case"], "repeat": trial["repeat"], **policies["candidate"]}
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phases", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.phases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified workload characteristics for {len(result['trials'])} paired trials")


if __name__ == "__main__":
    main()
