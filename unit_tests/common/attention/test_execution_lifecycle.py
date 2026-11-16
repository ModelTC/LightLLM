from types import SimpleNamespace


import pytest


import torch


from lightllm.common.execution import ExecutionScope


from lightllm.platform import DeviceRuntime


def test_submission_backpressure_retains_inputs_until_oldest_completion(monkeypatch):
    import gc
    import weakref

    runtime = DeviceRuntime("cpu")
    scope = ExecutionScope(runtime, max_pending=2)
    completed, events = [], []
    monkeypatch.setattr(runtime, "current_stream", lambda: "stream")

    def create_event():
        index = len(events)
        event = SimpleNamespace(
            record=lambda stream: None,
            query=lambda: index in completed,
            synchronize=lambda: completed.append(index),
        )
        events.append(event)
        return event

    monkeypatch.setattr(runtime, "create_event", create_event)
    source = torch.zeros(1)
    reference = weakref.ref(source)
    with scope.submit(retain=(source,)):
        pass
    del source
    with scope.submit():
        assert completed == []
        assert reference() is not None
    with scope.submit():
        gc.collect()
        assert completed == [0]
        assert reference() is None
    scope.synchronize()


def test_backpressure_failure_quarantines_without_releasing_inputs(monkeypatch):
    runtime = DeviceRuntime("cpu")
    scope = ExecutionScope(runtime, max_pending=1)
    source = object()
    monkeypatch.setattr(runtime, "current_stream", lambda: "stream")

    def fail():
        raise RuntimeError("device failed")

    monkeypatch.setattr(
        runtime,
        "create_event",
        lambda: SimpleNamespace(record=lambda stream: None, query=lambda: False, synchronize=fail),
    )
    with scope.submit(retain=(source,)):
        pass
    with pytest.raises(RuntimeError, match="device failed"):
        with scope.submit():
            pytest.fail("submitted after failed completion")
    assert scope._pending[0][1] == [source]
    with pytest.raises(RuntimeError, match="quarantined"):
        with scope.submit():
            pass
