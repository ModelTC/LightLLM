import copy
import json

import pytest

from test.benchmark.service.stream_timing import streaming_completion_times, summarize_streaming_times


def stream(text="hello"):
    origin = 10_000_000_000
    return dict(
        status=200,
        start_perf_ns=origin,
        metadata=dict(request_start_ns=1_000_000, request_end_ns=3_001_000_000, was_cancelled=False),
        responses=[
            dict(
                perf_ns=origin + 500_000_000,
                packets=[dict(name="data", value=json.dumps(dict(choices=[dict(text=text)])))],
            ),
            dict(
                perf_ns=origin + 1_000_000_000,
                packets=[dict(name="data", value=json.dumps(dict(choices=[dict(text=text)])))],
            ),
            dict(
                perf_ns=origin + 2_000_000_000,
                packets=[dict(name="data", value=json.dumps(dict(choices=[dict(text="")])))],
            ),
            dict(perf_ns=origin + 3_000_000_000, packets=[dict(name="data", value="[DONE]")]),
        ],
    )


def test_invisible_tokens_do_not_shorten_http_latency():
    assert streaming_completion_times(stream()) == dict(
        http_completion_ms=3000,
        first_visible_text_ms=500,
        last_visible_text_ms=1000,
        done_after_last_visible_text_ms=2000,
    )


def test_empty_text_retains_http_timing_without_invented_text_latency():
    empty = stream("")
    empty["metadata"]["request_end_ns"] += 1_000_000_000
    empty["responses"][-1]["perf_ns"] += 1_000_000_000
    result = summarize_streaming_times([stream(), empty])
    assert result["requests"] == 2 and result["empty_text_responses"] == 1
    assert result["http_completion_all_requests_ms"]["p95"] == 3950
    assert result["http_completion_content_requests_ms"]["p95"] == 3000
    assert result["last_visible_text_ms"]["p95"] == 1000
    assert summarize_streaming_times([empty])["last_visible_text_ms"] is None


@pytest.mark.parametrize(
    "failure",
    ["no_done", "duplicate_done", "after_done", "out_of_order", "clock", "cancelled", "http", "float_timestamp"],
)
def test_rejects_invalid_stream_boundaries(failure):
    record = stream()
    if failure == "no_done":
        record["responses"].pop()
    elif failure == "duplicate_done":
        record["responses"].append(copy.deepcopy(record["responses"][-1]))
    elif failure == "after_done":
        response = copy.deepcopy(record["responses"][0])
        response["perf_ns"] = record["responses"][-1]["perf_ns"]
        record["responses"].append(response)
    elif failure == "out_of_order":
        record["responses"][1]["perf_ns"] = record["responses"][0]["perf_ns"] - 1
    elif failure == "clock":
        record["metadata"]["request_end_ns"] += record["start_perf_ns"]
    elif failure == "cancelled":
        record["metadata"]["was_cancelled"] = True
    elif failure == "http":
        record["status"] = 500
    elif failure == "float_timestamp":
        record["responses"][0]["perf_ns"] = float(record["responses"][0]["perf_ns"])
    with pytest.raises(ValueError):
        streaming_completion_times(record)
