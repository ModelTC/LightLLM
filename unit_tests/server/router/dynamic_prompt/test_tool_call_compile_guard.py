import datetime
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/compile_guard.py"
spec = importlib.util.spec_from_file_location("tool_call_compile_guard", path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
START = datetime.datetime(2026, 10, 17, 14, 51, 48, tzinfo=datetime.timezone.utc).timestamp()
BEGIN = "2026-10-17 14:51:48 [TileLang:tilelang.jit.kernel:INFO] TileLang begins to compile kernel `kernel`\n"
END = "2026-10-17 14:52:01 [TileLang:tilelang.jit.kernel:INFO] TileLang completes to compile kernel `kernel`\n"


def trial(start, finish):
    return {
        "case": "trace",
        "repeat": 0,
        **{label: {"started_unix": start, "finished_unix": finish} for label in ("candidate", "control")},
    }


def test_overlapping_compilation_is_detected_in_either_policy():
    for label in ("candidate", "control"):
        logs = {"candidate": "", "control": ""}
        logs[label] = BEGIN + END
        assert guard.contaminated(guard.audit_trial(trial(START - 5, START + 30), logs))


def test_parallel_rank_messages_preserve_both_intervals_and_utc():
    events = guard.compilation_events(BEGIN + BEGIN + END + END)
    assert len(events) == 2
    assert all(e["started_unix"] == START and e["finished_unix_upper"] == START + 14 for e in events)


def test_compilation_before_replay_is_excluded_with_conservative_second_boundary():
    logs = {"candidate": BEGIN + END, "control": ""}
    assert guard.contaminated(guard.audit_trial(trial(START + 13.9, START + 20), logs))
    assert not guard.contaminated(guard.audit_trial(trial(START + 14, START + 20), logs))
    assert not guard.contaminated(guard.audit_trial(trial(START - 10, START), logs))


@pytest.mark.parametrize("text", [BEGIN, END, BEGIN.replace("2026-10-17", "unknown") + END])
def test_incomplete_or_unparseable_compilation_records_fail_closed(text):
    with pytest.raises(ValueError):
        guard.compilation_events(text)


def test_unrelated_autotuning_warning_is_not_a_compilation_event():
    assert guard.compilation_events("WARNING no kernel config; enable autotune\n") == []


def test_invalid_trial_clocks_are_rejected():
    with pytest.raises(ValueError, match="invalid replay clock"):
        guard.audit_trial(trial(1, 1), {"candidate": "", "control": ""})


@pytest.mark.parametrize("during_replay", [False, True])
def test_controlled_phase_recomputes_overlaps_from_logs_instead_of_trusting_empty_findings(tmp_path, during_replay):
    window = trial(START - 5, START + 30)
    policies = {label: {**window[label], "overlapping_compilations": []} for label in ("candidate", "control")}
    for name, data in {
        "progress.json": {"status": "complete", "paired_trials": 1},
        "protocol.json": {"return_mode": "continuation", "seeds": [42]},
        "summary.json": [{"seed": 42}],
        "jit-audit.json": {
            "status": "complete",
            "trials": [{"case": "continuation", "repeat": 0, "policies": policies}],
        },
    }.items():
        (tmp_path / name).write_text(json.dumps(data))
    (tmp_path / "candidate-server.log").write_text(BEGIN + END if during_replay else "")
    (tmp_path / "control-server.log").write_text("")
    assert guard.audit_phase(tmp_path)["contaminated_pairs"] == int(during_replay)
