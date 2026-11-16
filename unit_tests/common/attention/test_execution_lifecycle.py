from types import SimpleNamespace


import pytest


import torch


from lightllm.common.attention.api.graph import TensorCopyPlan, tensor_input_copy, validate_copy_plan


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


@pytest.mark.parametrize("invalid", ["conflicting", "partial", "source", "unchanged", "expanded"])
def test_replay_manifest_rejects_aliases_before_any_write(invalid):
    target = torch.zeros(8)
    source = torch.ones(8)
    copies = tensor_input_copy("first", target, source)
    if invalid == "conflicting":
        copies += tensor_input_copy("second", target, source.clone())
    elif invalid == "partial":
        copies += tensor_input_copy("second", target[1:], source[:-1])
    elif invalid == "source":
        copies += tensor_input_copy("second", torch.empty(8), target)
    elif invalid == "unchanged":
        copies += tensor_input_copy("second", target[:4], target[:4])
    else:
        copies = tensor_input_copy("expanded", target[:1].expand(8), source)
    with pytest.raises(ValueError, match="conflicting|overlap|overwrite"):
        validate_copy_plan(copies)
    assert target.eq(0).all()


def test_replay_manifest_coalesces_cumulative_offsets_and_prefix_views():
    target, source = torch.zeros(8), torch.arange(8, dtype=torch.float32)
    copies = tensor_input_copy("cumulative", target, source)
    copies += tensor_input_copy("start", target[:-1], source[:-1])
    copies += tensor_input_copy("duplicate", target, source)
    plan = validate_copy_plan(copies)
    assert len(plan) == 1
    for destination, incoming in plan:
        destination.copy_(incoming)
    torch.testing.assert_close(target, source)


@pytest.mark.parametrize("change", ["stride", "storage", "conflict"])
def test_compiled_manifest_revalidates_sources_and_destination_identity(change):
    target, source = torch.zeros(4, 4), torch.ones(4, 4)
    copies = [(target, source), (target, source)]
    plan = TensorCopyPlan([tensor for tensor, _ in copies])
    assert len(plan.validate(copies)) == 1
    if change == "stride":
        target.transpose_(0, 1)
    elif change == "storage":
        target.set_(torch.empty_like(target))
    else:
        copies[-1] = (target, source.clone())
    with pytest.raises(ValueError, match="destination"):
        plan.validate(copies)
