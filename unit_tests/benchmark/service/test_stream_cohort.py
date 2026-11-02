from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from test.benchmark.service.stream_cohort import StreamingWorkload, distribution, run_cohort, summarize_cohort
from test.benchmark.service.token_stream import request_stream


def observation(started, arrivals, ended, input_tokens=2, ids=None):
    gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
    return dict(
        token_ids=list(range(len(arrivals))) if ids is None else ids,
        metrics=dict(
            input_tokens=input_tokens,
            output_tokens=len(arrivals),
            ttft_s=arrivals[0] - started,
            tpot_s=sum(gaps) / len(gaps) if gaps else None,
            e2e_s=ended - started,
            receiver_itl_s=gaps,
        ),
        timeline=dict(started=started, first_token=arrivals[0], last_token=arrivals[-1], ended=ended),
    )


def test_overlapping_requests_use_elapsed_time_and_actual_token_counts():
    results = [observation(1, [1.5, 2.5, 4.5], 5), observation(2, [2.7], 4, input_tokens=4)]
    metrics = summarize_cohort(results)
    assert metrics["elapsed_s"] == 4
    assert metrics["output_tokens"] == 4 and metrics["output_tokens_per_s"] == 1
    assert metrics["input_tokens_per_s"] == 1.5
    assert metrics["maximum_client_inflight"] == 2
    assert metrics["ttft_s"]["p50"] == pytest.approx(0.6)
    assert metrics["e2e_s"]["p50"] == 3
    assert metrics["tpot_s"]["count"] == 1
    assert metrics["receiver_itl_s"]["p50"] == 1.5


def test_cache_aggregates_require_complete_observability():
    results = [observation(1, [1.5], 2), observation(2, [2.5], 3)]
    results[0]["metrics"]["cached_input_tokens"] = 1
    partial = summarize_cohort(results)
    assert partial["cache_reported_requests"] == 1
    assert partial["cached_input_tokens"] is None and partial["uncached_input_tokens"] is None
    results[1]["metrics"]["cached_input_tokens"] = 0
    complete = summarize_cohort(results)
    assert complete["cache_reported_requests"] == 2
    assert complete["cached_input_tokens"] == 1 and complete["uncached_input_tokens"] == 3


def test_bursts_and_single_tokens_keep_their_metric_meaning():
    metrics = summarize_cohort([observation(1, [2, 2, 3], 4)])
    assert metrics["receiver_zero_gap_fraction"] == 0.5
    metrics = summarize_cohort([observation(1, [2], 4)])
    assert metrics["tpot_s"] is None and metrics["receiver_itl_s"] is None
    assert metrics["receiver_zero_gap_fraction"] is None


def test_percentiles_are_interpolated_and_invalid_samples_rejected():
    metrics = distribution([4, 2, 1, 3])
    assert metrics["p50"] == 2.5 and metrics["p95"] == pytest.approx(3.85)
    assert distribution([]) is None
    for values in ([float("nan")], [float("inf")], [-1]):
        with pytest.raises(ValueError):
            distribution(values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("started", float("nan")),
        ("started", 3),
        ("ended", 1),
        ("first_token", 4),
        ("last_token", 1),
    ],
)
def test_invalid_timelines_cannot_produce_throughput(field, value):
    result = observation(1, [2, 3], 4)
    result["timeline"][field] = value
    with pytest.raises(ValueError):
        summarize_cohort([result])


@pytest.mark.parametrize(
    "field,value",
    [("output_tokens", 99), ("output_tokens", True), ("input_tokens", 0), ("receiver_itl_s", []), ("ttft_s", 0)],
)
def test_inconsistent_counts_or_timing_cannot_pass(field, value):
    result = observation(1, [2, 3], 4)
    result["metrics"][field] = value
    with pytest.raises(ValueError):
        summarize_cohort([result])


def test_reference_mismatch_invalidates_entire_cohort_and_keeps_evidence(tmp_path):
    tasks = [StreamingWorkload("ok", [11, 12], [7, 8]), StreamingWorkload("bad", [11, 12], [7, 8])]

    def request(task, path):
        ids = [7, 8] if task.name == "ok" else [7, 9]
        path.write_text("raw response")
        return observation(1, [2, 3], 4, ids=ids)

    report = run_cohort(tasks, 2, request, tmp_path)
    assert report["status"] == "failed" and report["metrics"] is None
    assert [record["status"] for record in report["records"]] == ["passed", "failed"]
    assert report["records"][1]["result"]["token_ids"] == [7, 9]
    assert (tmp_path / "000001.sse").read_text() == "raw response"


def test_transport_failure_cannot_be_hidden_by_successful_requests(tmp_path):
    task = StreamingWorkload("error", [11], [7])

    def request(task, path):
        path.write_text("partial response")
        raise TimeoutError("stream stalled")

    report = run_cohort([task], 1, request, tmp_path)
    assert report["status"] == "failed" and report["metrics"] is None
    assert report["records"][0]["error_type"] == "TimeoutError"


@pytest.mark.parametrize("concurrency", [0, -1, 2, True])
def test_invalid_concurrency_is_rejected(concurrency, tmp_path):
    with pytest.raises(ValueError):
        run_cohort([StreamingWorkload("x", [1], [2])], concurrency, lambda *_: None, tmp_path)


def test_real_chunked_sse_closed_loop_cohort(tmp_path):
    pair = threading.Barrier(2)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["prompt"] == [11, 12] and body["return_token_ids"] is True
            pair.wait(timeout=5)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            events = [
                {"choices": [{"index": 0, "token_ids": [7, 8]}]},
                {"choices": [{"index": 0, "token_ids": [9], "finish_reason": "length"}]},
                {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
            ]
            for payload in [*(json.dumps(event) for event in events), "[DONE]"]:
                chunk = ("data: " + payload + "\n\n").encode()
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    task = StreamingWorkload("fixture", [11, 12], [7, 8, 9])

    def request(task, path):
        return request_stream(
            f"http://127.0.0.1:{server.server_port}",
            "vllm",
            "fixture",
            list(task.prompt_ids),
            len(task.expected_ids),
            5,
            path,
        )

    try:
        report = run_cohort([task] * 4, 2, request, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert report["status"] == "passed"
    assert report["metrics"]["requests"] == 4
    assert report["metrics"]["output_tokens"] == 12
    assert report["metrics"]["maximum_client_inflight"] == 2
    assert len(list(tmp_path.glob("*.sse"))) == 4
