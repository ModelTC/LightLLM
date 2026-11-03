from unittest.mock import Mock

import pytest
import torch

from lightllm.utils import profiler


def test_timelog_samples_without_sync_and_limits_log_volume(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(profiler.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(profiler.time, "perf_counter", lambda: clock[0])
    events = []

    class Event:
        def __init__(self, **kwargs):
            self.record_count = 0
            events.append(self)

        def record(self):
            self.record_count += 1

        def elapsed_time(self, other):
            return 2.0

        def synchronize(self):
            raise AssertionError("timelog must reuse the caller synchronization")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    log = Mock()
    monkeypatch.setattr(profiler.logger, "info", log)
    timer = profiler.DecodeTimeLog(every=128, rank=9)
    for step in range(257):
        clock[0] = step / 2
        timer.begin()
        for name in [
            "round_start",
            "target_start",
            "target_end",
            "kv_commit_start",
            "kv_commit_end",
            "draft_start",
            "draft_end",
            "round_end",
        ]:
            timer.mark(name)
        timer.finish(requests=0, verify_rows=0, accepted_tokens=0)
    assert len(events) == 8
    assert {e.record_count for e in events} == {3}
    assert log.call_count == 2
    import json

    reports = [json.loads(call.args[1]) for call in log.call_args_list]
    assert [r["sampled_steps"] for r in reports] == [2, 1]
    assert reports[-1]["rank"] == 9
    assert reports[-1]["mean"]["target_ms"] == 2.0
    assert reports[-1]["mean"]["accepted_tokens"] == 0


def test_profiler_diagnostics_can_disable_expensive_capture(monkeypatch, tmp_path):
    monkeypatch.setenv("LIGHTLLM_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("LIGHTLLM_PROFILER_WITH_STACK", "0")
    monkeypatch.setenv("LIGHTLLM_PROFILER_RECORD_SHAPES", "0")
    p = profiler.ProcessProfiler(mode="torch_profiler")
    assert not p.torch_profiler_with_stack
    assert not p.torch_profiler_record_shapes
    record = Mock()
    monkeypatch.setattr(torch.profiler, "record_function", record)
    with p.phase("inactive"):
        pass
    record.assert_not_called()
    p.is_active = True
    p.phase("dspark.target_verify")
    record.assert_called_once_with("dspark.target_verify")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_timelog_can_measure_graph_replay_after_normal_round_sync(monkeypatch):
    monkeypatch.setattr(profiler.logger, "info", Mock())
    timer = profiler.DecodeTimeLog(every=128, rank=0)
    x = torch.ones(32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x + 1
    timer.begin()
    timer.mark("round_start")
    for phase in ["target", "kv_commit", "draft"]:
        timer.mark(phase + "_start")
        graph.replay()
        timer.mark(phase + "_end")
    timer.mark("round_end")
    # This represents the synchronization already required by decode_mtp.
    done = torch.cuda.Event()
    done.record()
    done.synchronize()
    timer.finish(requests=1, verify_rows=4, accepted_tokens=2)
    assert len(timer.samples) == 1
    assert timer.samples[0]["gpu_round_ms"] >= 0
    torch.testing.assert_close(y, x + 1)
