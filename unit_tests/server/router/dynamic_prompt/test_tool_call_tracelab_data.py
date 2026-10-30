import copy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/tracelab_data.py"
spec = importlib.util.spec_from_file_location("tool_call_tracelab_data", path)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def stamp(second):
    return f"2020-01-01T00:00:{second:02d}Z"


def row(index, start=None, **extra):
    start = index * 5 if start is None else start
    return {
        "provider": "p",
        "user": "private-user",
        "session_id": "private-session",
        "round_index": index,
        "input_tokens_total": 100 + (index or 0),
        "output_tokens": 10,
        "timing_events": [{"event_type": "text", "timestamp": stamp(start)}],
        "tools": [],
        **extra,
    }


def test_tool_result_readiness_and_identity_redaction():
    first = row(0, tools=[{"tool_name": "private-tool", "emitted_at": stamp(1), "result_at": stamp(3)}])
    rows, counts = data.normalize([first, row(1)])
    assert counts["selected_rows"] == 2
    assert rows[0]["next_input_ready_gap_s"] == 2
    assert rows[0]["next_first_chunk_gap_s"] == 4
    assert rows[0]["tool_names"] == ["observed_tool"]
    assert rows[0]["observed_finish_reason"] is None
    assert "next_first_chunk_gap_s" not in rows[-1]
    assert "private" not in repr(rows)


def test_missing_readiness_stays_missing():
    rows, _ = data.normalize([row(0), row(1)])
    assert rows[0]["next_input_ready_gap_s"] is None
    assert rows[0]["next_first_chunk_gap_s"] == 5


@pytest.mark.parametrize(
    "bad", [{"input_tokens_total": None}, {"output_tokens": None}, {"input_tokens_total": True}, {"timing_events": []}]
)
def test_invalid_round_splits_without_bridging(bad):
    rows, counts = data.normalize([row(0), row(1, **bad), row(2), row(3)])
    assert [r["round_index"] for r in rows] == [2, 3]
    assert counts["unselected_valid_rows"] == 1


@pytest.mark.parametrize("middle", [row(1, start=1), row(4)])
def test_overlap_or_index_gap_starts_a_new_segment(middle):
    first = row(0)
    first["timing_events"].append({"event_type": "text", "timestamp": stamp(2)})
    rows, counts = data.normalize([first, middle])
    assert len(rows) == 1
    assert rows[0]["round_index"] == 0
    assert counts["discontinuities"] == 1


def test_missing_index_rejects_session_without_bridging():
    rows, counts = data.normalize([row(0), row(None, start=5), row(2)])
    assert rows == []
    assert counts["sessions_rejected_missing_index"] == 1


def test_provider_and_user_are_part_of_session_identity():
    original = row(0)
    another = copy.deepcopy(original)
    another["user"] = "other"
    rows, counts = data.normalize([original, another])
    assert counts["selected_sessions"] == 2
    assert len({r["session"] for r in rows}) == 2


def test_only_observed_input_times_between_generations_count():
    following = row(1)
    following["timing_events"] += [
        {"event_type": "user_message", "timestamp": stamp(3)},
        {"event_type": "tool_result", "timestamp": stamp(9)},
        {"event_type": "user_message", "timestamp": "invalid"},
    ]
    rows, _ = data.normalize([row(0), following])
    assert rows[0]["next_input_ready_gap_s"] == 3


def test_usage_report_is_not_fabricated_generation_time():
    value, reason = data.normalize_row(row(0, timing_events=[{"event_type": "usage_report", "timestamp": stamp(0)}]))
    assert value is None
    assert reason == "missing_generation_timing"
