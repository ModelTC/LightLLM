import copy
import json

import pytest

from test.benchmark.qwen38.measure_aiperf import audit_records
from test.benchmark.qwen38.prepare_aiperf_trace import select_requests


def test_trace_selection_preserves_bursts_prefixes_and_lengths():
    rows = [
        dict(timestamp=0, input_length=4000, output_length=200, hash_ids=[1]),
        dict(timestamp=10, input_length=20, output_length=3, hash_ids=[2]),
        dict(timestamp=10, input_length=30, output_length=4, hash_ids=[2, 3]),
        dict(timestamp=100, input_length=10, output_length=5, hash_ids=[4]),
    ]
    selected, count = select_requests(rows, 2, 4096)
    assert count == 3 and selected == [(1, rows[1]), (2, rows[2])]
    with pytest.raises(ValueError, match="Not enough"):
        select_requests(rows, 4, 4096)
    with pytest.raises(ValueError, match="nondecreasing"):
        select_requests(list(reversed(rows)), 1, 4096)


def fixture_record():
    return dict(
        metadata=dict(
            benchmark_phase="profiling",
            conversation_id="session_000000",
            session_num=0,
            request_start_ns=1000,
            was_cancelled=False,
        ),
        status=200,
        payload=dict(prompt=["hello"], max_tokens=2, temperature=0, ignore_eos=True),
        responses=[
            dict(
                packets=[
                    dict(
                        name="data",
                        value=json.dumps(
                            dict(
                                choices=[dict(text="hello", finish_reason="length")],
                                usage=dict(prompt_tokens=1, completion_tokens=2),
                            )
                        ),
                    ),
                    dict(name="data", value="[DONE]"),
                ]
            )
        ],
    )


def write_artifacts(path, records):
    metrics = (
        "request_throughput",
        "output_token_throughput",
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
    )
    summary = dict(error_summary=[], **{name: dict(avg=1) for name in metrics})
    summary.update(request_count=dict(avg=len(records)), output_token_count=dict(sum=2 * len(records)))
    (path / "profile_export_aiperf.json").write_text(json.dumps(summary))
    (path / "profile_export_raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))


def test_raw_audit_accepts_complete_stream_and_excludes_model_alias(tmp_path):
    row = fixture_record()
    write_artifacts(tmp_path, [row])
    trace = [dict(timestamp=0, output_length=2)]
    result = audit_records(tmp_path, trace)
    row["payload"]["model"] = "different-service-alias"
    write_artifacts(tmp_path, [row])
    assert audit_records(tmp_path, trace)["request_manifest_sha256"] == result["request_manifest_sha256"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "cancelled", "http", "no_done", "count"])
def test_raw_audit_rejects_invalid_work(tmp_path, failure):
    row = fixture_record()
    rows = [row]
    if failure == "missing":
        rows = []
    elif failure == "duplicate":
        rows.append(copy.deepcopy(row))
    elif failure == "cancelled":
        row["metadata"]["was_cancelled"] = True
    elif failure == "http":
        row["status"] = 500
    elif failure == "no_done":
        row["responses"][0]["packets"].pop()
    elif failure == "count":
        packet = row["responses"][0]["packets"][0]
        body = json.loads(packet["value"])
        body["usage"]["completion_tokens"] = 1
        packet["value"] = json.dumps(body)
    write_artifacts(tmp_path, rows)
    with pytest.raises(ValueError):
        audit_records(tmp_path, [dict(timestamp=0, output_length=2)])


@pytest.mark.parametrize("error_type", ["InvalidInferenceResultError", "TimeoutError"])
def test_complete_empty_text_stays_an_explicit_error(tmp_path, error_type):
    empty, valid = fixture_record(), fixture_record()
    valid["metadata"].update(session_num=1, conversation_id="session_000001", request_start_ns=2000)
    packet = empty["responses"][0]["packets"][0]
    body = json.loads(packet["value"])
    body["choices"][0]["text"] = ""
    packet["value"] = json.dumps(body)
    write_artifacts(tmp_path, [empty, valid])
    path = tmp_path / "profile_export_aiperf.json"
    summary = json.loads(path.read_text())
    summary.update(
        request_count=dict(avg=1),
        output_token_count=dict(sum=2),
        error_summary=[dict(count=1, error_details=dict(type=error_type, message="No responses with actual content"))],
    )
    path.write_text(json.dumps(summary))
    trace = [dict(timestamp=0, output_length=2), dict(timestamp=1, output_length=2)]
    if error_type == "TimeoutError":
        with pytest.raises(ValueError, match="other request error"):
            audit_records(tmp_path, trace)
    else:
        result = audit_records(tmp_path, trace)
        assert result["error_rate"] == 0.5 and result["empty_content_sessions"] == [0]
        assert result["output_tokens"] == 4 and result["successful_output_tokens"] == 2


def test_concurrent_dispatch_order_does_not_reorder_source_requests(tmp_path):
    first, second = fixture_record(), fixture_record()
    second["metadata"].update(session_num=1, conversation_id="session_000001", request_start_ns=2000)
    second["payload"]["prompt"] = ["different"]
    trace = [dict(timestamp=0, output_length=2), dict(timestamp=1, output_length=2)]
    write_artifacts(tmp_path, [first, second])
    reference = audit_records(tmp_path, trace)
    first["metadata"]["session_num"], second["metadata"]["session_num"] = 1, 0
    write_artifacts(tmp_path, [second, first])
    assert audit_records(tmp_path, trace) == reference
