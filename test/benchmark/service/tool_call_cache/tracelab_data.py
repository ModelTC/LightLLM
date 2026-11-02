"""Normalize pinned public TraceLab metadata for independent trace-shaped replay.

No prompts or observed finish reasons are available. Tool presence stays an
advisory label, and input readiness is recorded only when observed. Keep one
longest contiguous, nonoverlapping segment per original session, with a stable
first-segment tie break; never join across missing usage or invalid rounds.
"""

import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b"
GENERATION_EVENTS = {"reasoning", "text", "tool_call"}
INPUT_EVENTS = {"user_message", "tool_result"}


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.timestamp() if result.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def observed_times(row, kinds):
    return [
        value
        for event in row.get("timing_events", []) or []
        if event.get("event_type") in kinds
        for value in [timestamp(event.get("timestamp"))]
        if value is not None
    ]


def normalize_row(row):
    inputs, outputs, index = row.get("input_tokens_total"), row.get("output_tokens"), row.get("round_index")
    if any(type(value) is not int for value in (inputs, outputs, index)) or inputs <= 0 or outputs < 0:
        return None, "invalid_usage_or_index"
    generated = observed_times(row, GENERATION_EVENTS)
    tools = row.get("tools", []) or []
    generated += [value for tool in tools for value in [timestamp(tool.get("emitted_at"))] if value is not None]
    if not generated:
        return None, "missing_generation_timing"
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        # Presence alone drives the synthetic echo tool; source names/IDs are
        # unnecessary and are deliberately excluded from the normalized file.
        "tool_names": ["observed_tool"] if tools else [],
        "observed_finish_reason": None,
        "first_chunk_s": min(generated),
        "last_chunk_s": max(generated),
        "round_index": index,
        "_input_times": observed_times(row, INPUT_EVENTS),
        "_result_times": [value for tool in tools for value in [timestamp(tool.get("result_at"))] if value is not None],
    }, None


def normalize(rows):
    grouped = collections.defaultdict(list)
    counts = collections.Counter()
    for row in rows:
        counts["source_rows"] += 1
        identity = (row.get("provider"), row.get("user"), row.get("session_id"))
        if identity[0] is None or identity[2] is None:
            counts["missing_session_identity"] += 1
            continue
        grouped[identity].append(row)
    output = []
    for identity, source in sorted(grouped.items(), key=lambda item: json.dumps(item[0])):
        segments, current = [], []
        # Rows with missing indices cannot be placed; reject the whole session
        # rather than silently bridge a potentially missing request.
        if any(type(row.get("round_index")) is not int for row in source):
            counts["sessions_rejected_missing_index"] += 1
            continue
        for row in sorted(source, key=lambda item: item["round_index"]):
            item, reason = normalize_row(row)
            if item is None:
                counts[reason] += 1
                if current:
                    segments.append(current)
                    current = []
                continue
            if current and (
                item["round_index"] != current[-1]["round_index"] + 1
                or item["first_chunk_s"] < current[-1]["last_chunk_s"]
            ):
                counts["discontinuities"] += 1
                segments.append(current)
                current = []
            current.append(item)
        if current:
            segments.append(current)
        if not segments:
            continue
        selected = max(segments, key=len)
        counts["unselected_valid_rows"] += sum(map(len, segments)) - len(selected)
        for previous, following in zip(selected, selected[1:]):
            first, last = following["first_chunk_s"], previous["last_chunk_s"]
            ready = [value for value in previous["_result_times"] + following["_input_times"] if last <= value <= first]
            previous["next_first_chunk_gap_s"] = first - last
            previous["next_input_ready_gap_s"] = max(ready) - last if ready else None
            previous["next_round_overlaps"] = False
        session = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).hexdigest()
        for item in selected:
            item.pop("_input_times")
            item.pop("_result_times")
            item["session"] = session
            item["request"] = hashlib.sha256(f'{session}/{item["round_index"]}'.encode()).hexdigest()
            output.append(item)
        counts["selected_sessions"] += 1
        counts["eligible_sessions_8_turns"] += len(selected) >= 8
    counts["selected_rows"] = len(output)
    return output, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        parser.error("source does not match the pinned TraceLab v0.0.1 release")
    rows = (json.loads(line) for line in gzip.decompress(payload).splitlines() if line)
    normalized, counts = normalize(rows)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized))
    provenance = {
        "source_sha256": SOURCE_SHA256,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "counts": counts,
        "source": "https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1",
        "license": "CC BY 4.0",
        "selection": "One longest contiguous nonoverlapping valid segment per original session; first wins ties.",
        "timing": "Generation events bound output; observed result/user times bound input readiness; otherwise null.",
        "limitations": (
            "Metadata only, synthetic contents; missing finish reasons remain null; "
            "provider prefix billing is unused."
        ),
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
