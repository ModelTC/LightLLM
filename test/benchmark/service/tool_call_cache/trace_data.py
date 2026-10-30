"""Download pinned public coding traces and inspect observed continuation timing.

Raw Claude Code logs contain multiple records per generation. Group by request
ID before counting rounds. Missing stop reasons stay missing: a tool_use content
block supplies a separate advisory label, not a fabricated finish_reason.
"""

import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request


REPOSITORY = "cachewise-project/cachewise-coding-traces"


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download(directory, revision):
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/"
    raw = fetch(base + "raw_traces/manifest.json")
    (directory / "manifest.json").write_bytes(raw)
    manifest = json.loads(raw)

    def one(session):
        relative = session["output_path"].removeprefix("release/")
        payload = fetch(base + relative)
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(one, manifest["sessions"]))
    provenance = {"repository": REPOSITORY, "revision": revision, "files": files}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"downloaded_sessions": len(files), "bytes": sum(f["bytes"] for f in files)}))


def timestamp(value):
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def extract_rounds(rows):
    rows = [r for r in rows if not r.get("isSidechain", False)]
    grouped = collections.OrderedDict()
    for index, row in enumerate(rows):
        message = row.get("message", {})
        if row.get("type") != "assistant":
            continue
        request = row.get("requestId") or message.get("id")
        if request is None:
            continue
        group = grouped.setdefault(request, [])
        group.append((index, row))
    rounds = []
    for request, chunks in grouped.items():
        messages = [row["message"] for _, row in chunks]
        usages = [m.get("usage", {}) for m in messages]
        input_tokens = max(
            (
                sum(
                    u.get(k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
                )
                for u in usages
            ),
            default=0,
        )
        tools = [
            block.get("name")
            for m in messages
            for block in m.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        reasons = [m["stop_reason"] for m in messages if m.get("stop_reason") is not None]
        rounds.append(
            {
                "request": request,
                "session": chunks[0][1].get("sessionId"),
                "input_tokens": input_tokens,
                "output_tokens": max((u.get("output_tokens", 0) or 0 for u in usages), default=0),
                "tool_names": tools,
                "observed_finish_reason": reasons[-1] if reasons else None,
                "first_chunk_s": min(timestamp(r["timestamp"]) for _, r in chunks),
                "last_chunk_s": max(timestamp(r["timestamp"]) for _, r in chunks),
                "first_index": chunks[0][0],
                "last_index": chunks[-1][0],
            }
        )
    for current, following in zip(rounds, rounds[1:]):
        current["next_round_overlaps"] = following["first_chunk_s"] < current["last_chunk_s"]
        current["next_first_chunk_gap_s"] = max(0, following["first_chunk_s"] - current["last_chunk_s"])
        inputs = [r for r in rows[current["last_index"] + 1 : following["first_index"]] if r.get("type") == "user"]
        # This measures input readiness, not actual HTTP arrival. The next first
        # chunk gives a second, looser bound including server/transport latency.
        current["next_input_ready_gap_s"] = (
            max(0, max(timestamp(r["timestamp"]) for r in inputs) - current["last_chunk_s"]) if inputs else None
        )
    return rounds


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        **{f"p{p}": values[min(len(values) - 1, int((len(values) - 1) * p / 100))] for p in [50, 90, 95, 99]},
    }


def observed_finish_reason_analysis(rounds):
    """Analyze original labels separately; never fill missing ones from content."""
    groups = {}
    for label, reasons in [
        ("tool_use", {"tool_use"}),
        ("end_turn", {"end_turn"}),
        ("stop_sequence", {"stop_sequence"}),
        ("length_known", {"max_tokens", "length"}),
        ("ordinary_known", {"end_turn", "stop_sequence"}),
    ]:
        selected = [row for row in rounds if row.get("observed_finish_reason") in reasons]
        gaps = [
            row["next_first_chunk_gap_s"]
            for row in selected
            if "next_first_chunk_gap_s" in row and not row.get("next_round_overlaps", False)
        ]
        ready = [
            row["next_input_ready_gap_s"]
            for row in selected
            if row.get("next_input_ready_gap_s") is not None and not row.get("next_round_overlaps", False)
        ]
        groups[label] = {
            "rounds": len(selected),
            "sessions": len({row["session"] for row in selected}),
            "no_observed_followup": sum("next_first_chunk_gap_s" not in row for row in selected),
            "overlaps_excluded": sum(row.get("next_round_overlaps", False) for row in selected),
            "first_chunk_gap_seconds": distribution(gaps),
            "input_readiness_gap_seconds": distribution(ready),
            "fraction_within_seconds_given_observed_followup": {
                str(t): sum(g <= t for g in gaps) / len(gaps) if gaps else None for t in [1, 5, 10, 30, 60, 300]
            },
            "input_ready_fraction_within_seconds_given_observed_ready": {
                str(t): sum(g <= t for g in ready) / len(ready) if ready else None for t in [1, 5, 10, 30, 60, 300]
            },
        }
    return {
        "all_rounds": len(rounds),
        "missing_original_finish_reason": sum(row.get("observed_finish_reason") is None for row in rounds),
        "groups": groups,
        "limitations": [
            "Most original reasons are missing; missingness is not assumed random.",
            "Final observations are censored; fractions are conditional on observed nonoverlapping followups.",
            "Input readiness and first response chunks are different bounds, not exact HTTP arrivals.",
            "Length terminations are reported separately; the data do not justify treating them as end_turn.",
        ],
    }


def analyze(directory, output):
    provenance = json.loads((directory / "provenance.json").read_text())
    rounds = []
    for file in provenance["files"]:
        data = (directory / file["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != file["sha256"]:
            raise ValueError(f"trace checksum mismatch: {file['path']}")
        rounds.extend(extract_rounds([json.loads(line) for line in data.splitlines() if line]))
    groups = {}
    for name, tool in [("tool_content_observed", True), ("no_tool_content_observed", False)]:
        selected = [r for r in rounds if bool(r["tool_names"]) == tool]
        gaps = [
            r["next_first_chunk_gap_s"]
            for r in selected
            if "next_first_chunk_gap_s" in r and not r["next_round_overlaps"]
        ]
        ready = [r["next_input_ready_gap_s"] for r in selected if r.get("next_input_ready_gap_s") is not None]
        groups[name] = {
            "rounds": len(selected),
            "no_observed_followup": sum("next_first_chunk_gap_s" not in r for r in selected),
            "overlapping_followup_excluded": sum(r.get("next_round_overlaps", False) for r in selected),
            "next_first_chunk_gap_s": distribution(gaps),
            "next_input_ready_gap_s": distribution(ready),
            "fraction_within_s_given_observed_followup": {
                str(t): sum(g <= t for g in gaps) / len(gaps) if gaps else None for t in [1, 5, 10, 30, 60, 300]
            },
        }
    result = {
        "source": {k: provenance[k] for k in ["repository", "revision"]},
        "sessions": len(provenance["files"]),
        "sessions_with_main_thread_rounds": len({r["session"] for r in rounds}),
        "rounds": len(rounds),
        "missing_original_finish_reason": sum(r["observed_finish_reason"] is None for r in rounds),
        "input_tokens": distribution([r["input_tokens"] for r in rounds]),
        "groups": groups,
        "original_finish_reason_analysis": observed_finish_reason_analysis(rounds),
        "limitations": [
            "Tool-content labels are inferred independently of original API finish reasons.",
            "Final observations are censored; missing followup does not prove no future request.",
            "Input readiness and first response chunk include orchestration and server latency; "
            "exact HTTP arrivals are unavailable.",
            "Public text is redacted, so token identity cannot be reconstructed exactly.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.with_suffix(".rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download", "analyze"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "download":
        if args.revision is None or len(args.revision) != 40:
            parser.error("download requires a full pinned --revision SHA")
        download(args.directory, args.revision)
    else:
        if args.output is None:
            parser.error("analyze requires --output")
        analyze(args.directory, args.output)


if __name__ == "__main__":
    main()
