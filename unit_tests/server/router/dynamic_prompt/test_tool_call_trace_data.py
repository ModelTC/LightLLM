import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/trace_data.py"
spec = importlib.util.spec_from_file_location("tool_call_trace_data", path)
trace_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_data)


def row(kind, second, request=None, content=None, **extra):
    return {
        "type": kind,
        "timestamp": f"2020-01-01T00:00:0{second}Z",
        "sessionId": "s1",
        "requestId": request,
        "message": {
            "content": content or [],
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
                "output_tokens": second,
            },
            "stop_reason": None,
        },
        **extra,
    }


def test_stream_chunks_are_one_round_and_final_observation_is_censored():
    rows = [
        row("user", 0),
        row("assistant", 1, "r1", [{"type": "text", "text": "redacted"}]),
        row("assistant", 2, "r1", [{"type": "tool_use", "name": "Grep"}]),
        row("user", 3, content=[{"type": "tool_result", "tool_use_id": "t1"}]),
        row("assistant", 4, "r2"),
    ]
    rounds = trace_data.extract_rounds(rows)
    assert len(rounds) == 2
    assert rounds[0]["input_tokens"] == 31
    assert rounds[0]["output_tokens"] == 2
    assert rounds[0]["tool_names"] == ["Grep"]
    assert rounds[0]["observed_finish_reason"] is None
    assert rounds[0]["next_input_ready_gap_s"] == 1
    assert rounds[0]["next_first_chunk_gap_s"] == 2
    assert not rounds[0]["next_round_overlaps"]
    assert "next_first_chunk_gap_s" not in rounds[-1]


def test_sidechains_are_excluded_and_original_reason_is_preserved():
    original = row("assistant", 2, "r1")
    original["message"]["stop_reason"] = "max_tokens"
    rows = [row("assistant", 1, "side", isSidechain=True), original]
    rounds = trace_data.extract_rounds(rows)
    assert len(rounds) == 1
    assert rounds[0]["observed_finish_reason"] == "max_tokens"


def test_overlapping_rounds_are_explicitly_marked():
    rows = [row("assistant", 1, "r1"), row("assistant", 2, "r2"), row("assistant", 3, "r1")]
    rounds = trace_data.extract_rounds(rows)
    assert rounds[0]["next_round_overlaps"]


def test_direct_finish_reason_analysis_preserves_missing_labels_and_fast_length_followups():
    def observation(reason, gap, **extra):
        return {
            "observed_finish_reason": reason,
            "session": "s",
            "next_first_chunk_gap_s": gap,
            "next_input_ready_gap_s": gap,
            **extra,
        }

    rows = [
        observation("tool_use", 0.2),
        observation("tool_use", 0, next_round_overlaps=True),
        observation("end_turn", 90),
        observation("max_tokens", 0.5),
        observation(None, 0.01, tool_names=["Bash"]),
    ]
    result = trace_data.observed_finish_reason_analysis(rows)
    assert result["missing_original_finish_reason"] == 1
    assert result["groups"]["tool_use"]["rounds"] == 2
    assert result["groups"]["tool_use"]["first_chunk_gap_seconds"]["n"] == 1
    assert result["groups"]["tool_use"]["input_readiness_gap_seconds"]["p50"] == 0.2
    assert result["groups"]["length_known"]["input_readiness_gap_seconds"]["p50"] == 0.5
    assert result["groups"]["ordinary_known"]["rounds"] == 1
