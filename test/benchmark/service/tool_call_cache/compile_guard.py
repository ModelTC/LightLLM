"""Reject logged TileLang compilation overlapping measured replay intervals."""

import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import re


PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[TileLang:.*?"
    r"TileLang (begins|completes) to compile kernel `([A-Za-z0-9_]+)`"
)


def compilation_events(text):
    pending = collections.defaultdict(collections.deque)
    events = []
    for line in text.splitlines():
        if "TileLang" not in line or "to compile kernel" not in line:
            continue
        match = PATTERN.search(line)
        if match is None:
            raise ValueError("cannot parse TileLang compilation timestamp")
        stamp, action, kernel = match.groups()
        instant = (
            datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
        )
        if action == "begins":
            pending[kernel].append(instant)
        else:
            if not pending[kernel]:
                raise ValueError("TileLang compilation completion has no matching start")
            started = pending[kernel].popleft()
            if instant < started:
                raise ValueError("TileLang compilation clock moved backwards")
            # Logger timestamps have one-second precision. Keep the entire
            # last second rather than undercount an overlap at its boundary.
            events.append({"kernel": kernel, "started_unix": started, "finished_unix_upper": instant + 1})
    if any(pending.values()):
        raise ValueError("TileLang compilation has not completed in captured logs")
    return events


def audit_trial(trial, logs):
    result = {"case": trial["case"], "repeat": trial["repeat"], "policies": {}}
    for label in ("candidate", "control"):
        started, finished = (trial[label][key] for key in ("started_unix", "finished_unix"))
        if not all(math.isfinite(value) for value in (started, finished)) or finished <= started:
            raise ValueError("invalid replay clock interval")
        overlaps = [
            event
            for event in compilation_events(logs[label])
            if event["started_unix"] < finished and event["finished_unix_upper"] > started
        ]
        result["policies"][label] = {
            "started_unix": started,
            "finished_unix": finished,
            "overlapping_compilations": overlaps,
        }
    return result


def contaminated(audit):
    return any(policy["overlapping_compilations"] for policy in audit["policies"].values())


def audit_phase(phase):
    progress = json.loads((phase / "progress.json").read_text())
    protocol = json.loads((phase / "protocol.json").read_text())
    trials = json.loads((phase / "summary.json").read_text())
    controlled = protocol.get("return_mode") == "continuation"
    expected = len(protocol["seeds"]) if controlled else len(protocol["cases"]) * protocol["repeats"]
    if progress.get("status") != "complete" or len(trials) != expected:
        raise ValueError("compilation audit requires a complete phase")
    if controlled:
        timing = json.loads((phase / "jit-audit.json").read_text())
        if timing.get("status") != "complete" or len(timing["trials"]) != expected:
            raise ValueError("controlled compilation audit requires all wrapper intervals")
        if [trial["seed"] for trial in trials] != protocol["seeds"]:
            raise ValueError("controlled compilation audit seed coverage differs")
        windows = []
        for repeat, trial in enumerate(timing["trials"]):
            if trial["case"] != "continuation" or trial["repeat"] != repeat:
                raise ValueError("controlled compilation audit trial coverage differs")
            if set(trial["policies"]) != {"candidate", "control"}:
                raise ValueError("controlled compilation audit omits a policy")
            windows.append({"case": "continuation", "repeat": repeat, **trial["policies"]})
    else:
        windows = trials
    logs = {label: (phase / f"{label}-server.log").read_text() for label in ("candidate", "control")}
    audits = [audit_trial(trial, logs) for trial in windows]
    return {
        "scope": "Logged TileLang compilation overlap only; this diagnostic does not waive phase invalidity.",
        "paired_trials": len(audits),
        "contaminated_pairs": sum(contaminated(audit) for audit in audits),
        "server_log_sha256": {label: hashlib.sha256(text.encode()).hexdigest() for label, text in logs.items()},
        "trials": audits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_phase(args.phase)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in ("paired_trials", "contaminated_pairs")})
    if result["contaminated_pairs"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
