import json
import math


def _stats(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return dict(count=len(ordered), p50=percentile(0.5), p95=percentile(0.95), p99=percentile(0.99), max=ordered[-1])


def _nanoseconds(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer timestamp in nanoseconds")
    return value


def streaming_completion_times(record):
    metadata = record["metadata"]
    if record["status"] != 200 or metadata.get("was_cancelled"):
        raise ValueError("Cannot measure a failed or cancelled streaming response")
    start = _nanoseconds(metadata["request_start_ns"])
    end = _nanoseconds(metadata["request_end_ns"])
    origin = _nanoseconds(record["start_perf_ns"])
    if end < start:
        raise ValueError("HTTP end precedes request start")
    text_times, done_times = [], []
    previous = origin
    for response in record["responses"]:
        timestamp = _nanoseconds(response["perf_ns"])
        if timestamp < previous:
            raise ValueError("Nonmonotonic streaming packet timestamps")
        previous = timestamp
        for packet in response.get("packets", []):
            if packet.get("name") != "data":
                continue
            if done_times:
                raise ValueError("Unexpected SSE data after DONE")
            if packet["value"] == "[DONE]":
                done_times.append(timestamp)
                continue
            payload = json.loads(packet["value"])
            if any(choice.get("text") for choice in payload.get("choices", [])):
                text_times.append(timestamp)
    if len(done_times) != 1:
        raise ValueError("Expected one complete SSE DONE marker")

    if abs((done_times[0] - origin) - (end - start)) > 1_000_000:
        raise ValueError("HTTP completion and SSE DONE elapsed times disagree")
    completion = (end - start) / 1e6
    first = (text_times[0] - origin) / 1e6 if text_times else None
    last = (text_times[-1] - origin) / 1e6 if text_times else None
    return dict(
        http_completion_ms=completion,
        first_visible_text_ms=first,
        last_visible_text_ms=last,
        done_after_last_visible_text_ms=(done_times[0] - text_times[-1]) / 1e6 if text_times else None,
    )


def summarize_streaming_times(records):
    rows = [streaming_completion_times(record) for record in records]
    if not rows:
        raise ValueError("No streaming records to summarize")
    content = [row for row in rows if row["last_visible_text_ms"] is not None]
    return dict(
        requests=len(rows),
        empty_text_responses=len(rows) - len(content),
        http_completion_all_requests_ms=_stats([row["http_completion_ms"] for row in rows]),
        http_completion_content_requests_ms=_stats([row["http_completion_ms"] for row in content]),
        first_visible_text_ms=_stats([row["first_visible_text_ms"] for row in content]),
        last_visible_text_ms=_stats([row["last_visible_text_ms"] for row in content]),
        done_after_last_visible_text_ms=_stats([row["done_after_last_visible_text_ms"] for row in content]),
    )
