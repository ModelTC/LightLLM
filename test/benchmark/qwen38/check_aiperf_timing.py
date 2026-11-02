import argparse
import hashlib
import json
import os
from pathlib import Path

from test.benchmark.qwen38.measure_aiperf import audit_records, profiling_records
from test.benchmark.service.stream_timing import summarize_streaming_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="Completed aiperf-replay.json")
    args = parser.parse_args()
    if not os.environ.get("EXP_ACTIVE"):
        parser.error('Run through exp -m "purpose"')
    replay = json.loads(args.replay.read_text())
    if replay["status"] != "aiperf_replay_recorded":
        raise ValueError("A completed replay is required")
    directory = args.replay.parent
    trace_path = directory / "trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    audit = audit_records(directory / "aiperf", trace)
    if audit != replay["audit"]:
        raise ValueError("Retained replay audit differs from its raw records")
    timing = summarize_streaming_times(profiling_records(directory / "aiperf"))
    if timing["requests"] != audit["requests"] or timing["empty_text_responses"] != audit["content_errors"]:
        raise ValueError("Timing cohort differs from the complete replay")
    service = replay["service"]
    for key in ("framework", "version", "software", "hardware", "command", "configuration", "model_revision"):
        if not service.get(key):
            raise ValueError(f"Missing actual service provenance: {key}")
    paths = (
        args.replay,
        trace_path,
        directory / "aiperf/profile_export_raw.jsonl",
        Path(__file__),
        Path(__file__).parent.parent / "service/stream_timing.py",
    )
    report = dict(
        status="complete_aiperf_stream_timing_verified",
        replay=str(args.replay.resolve()),
        service=service,
        timing=timing,
        aiperf_metrics=audit["metrics"],
        hashes={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        comparison_eligible=False,
        limitations=(
            "Timing only. Complete work and explicit empty-text errors are retained. "
            "HTTP completion includes SSE DONE; visible-text boundaries exclude later invisible-token generation. "
            "Task quality and matched framework comparisons require separate audits."
        ),
    )
    run = Path(os.environ["EXP_RUN_DIR"])
    (run / "stream-timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], timing=timing)))


if __name__ == "__main__":
    main()
