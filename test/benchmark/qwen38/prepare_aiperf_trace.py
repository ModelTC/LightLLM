import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


REVISION = "3cca71daccf2a7afb8fe3f0295358f70e3a69fdb"
URL = f"https://raw.githubusercontent.com/kvcache-ai/Mooncake/{REVISION}/FAST25-release/traces/conversation_trace.jsonl"
SHA256 = "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df"


def select_requests(rows, count, context_length):
    if count < 1 or context_length < 2:
        raise ValueError("Positive count and context length >= 2 are required")
    eligible = []
    previous = -1
    for index, row in enumerate(rows):
        if row["timestamp"] < previous:
            raise ValueError("Trace timestamps must be nondecreasing")
        previous = row["timestamp"]
        if (
            row["input_length"] > 0
            and row["output_length"] > 0
            and row["input_length"] + row["output_length"] <= context_length
        ):
            eligible.append((index, row))
    if len(eligible) < count:
        raise ValueError("Not enough requests fit the shared context limit")
    return eligible[:count], len(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=4096)
    args = parser.parse_args()
    raw = args.source.read_bytes() if args.source else urlopen(URL, timeout=60).read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Public trace differs from the pinned source")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected, eligible = select_requests(rows, args.count, args.context_length)
    directory = args.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "conversation_trace.original.jsonl").write_bytes(raw)
    trace = directory / "replay.jsonl"
    trace.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for _, row in selected))
    report = dict(
        source_url=URL,
        revision=REVISION,
        source_sha256=SHA256,
        source_requests=len(rows),
        eligible_requests=eligible,
        selected_indices=[index for index, _ in selected],
        requests=len(selected),
        context_length=args.context_length,
        selection="First eligible requests in source order; no input/output clipping",
        prompt_content="Anonymized lengths and 512-token prefix hashes; AIPerf synthesizes text",
        timing="Original millisecond timestamps, normalized by AIPerf; speed multipliers reported separately",
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        schedule_span_ms=selected[-1][1]["timestamp"] - selected[0][1]["timestamp"],
        input_tokens=sum(row["input_length"] for _, row in selected),
        output_tokens=sum(row["output_length"] for _, row in selected),
    )
    (directory / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
