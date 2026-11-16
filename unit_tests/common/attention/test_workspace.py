import pytest
import torch

from lightllm.common.attention.api.workspace import AttentionWorkspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_workspace_cross_stream_reuse_waits_for_writer():
    workspace = AttentionWorkspace("cuda")
    buffer = workspace.scratch("shared", 4096, torch.int32)
    buffer.zero_()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        with workspace.execution.submit():
            torch.cuda._sleep(2_000_000)
            buffer.fill_(17)
    with torch.cuda.stream(consumer):
        with workspace.execution.submit():
            result = buffer.clone()
    consumer.synchronize()
    assert result.eq(17).all()

    larger = workspace.scratch("shared", 8192, torch.int32)
    assert larger.data_ptr() != buffer.data_ptr()
    assert buffer.eq(17).all()


def test_capture_owns_provider_objects_and_preserves_frozen_addresses():
    workspace = AttentionWorkspace("cpu")
    fork = workspace.borrow_for_capture()
    plan = fork.get_object("provider_plan", dict)
    assert workspace.get_object("provider_plan", dict) is not plan
    assert workspace.borrow_for_capture().get_object("provider_plan", dict) is not plan
    with pytest.raises(TypeError, match="provider_plan requires list"):
        workspace.get_object("provider_plan", list)
    with pytest.raises(TypeError, match="bad requires dict"):
        workspace.get_object("bad", lambda: [], expected_type=dict)
    assert "bad" not in workspace._objects
    buffer = fork.scratch("scratch", 4)
    fork.freeze()
    workspace.scratch("scratch", 8)
    assert fork.scratch("scratch", 4) is buffer
    with pytest.raises(RuntimeError, match="cannot grow"):
        fork.scratch("scratch", 8)
    with pytest.raises(RuntimeError, match="before capture"):
        fork.get_object("next_bucket", dict)
    workspace.get_object("next_bucket", dict)
    with pytest.raises(RuntimeError, match="before capture"):
        fork.get_object("next_bucket", dict)


def test_cpu_stream_and_event_have_consistent_noop_lifecycle():
    workspace = AttentionWorkspace("cpu")
    runtime = workspace.runtime
    assert runtime.create_stream() is None
    assert runtime.create_event() is None
    with runtime.use_stream(runtime.create_stream()):
        with workspace.execution.submit():
            pass


def test_pool_and_capture_reject_another_host_owner():
    pool = AttentionWorkspace("cpu")
    capture = pool.borrow_for_capture()
    with pool.execution.submit():
        with pytest.raises(RuntimeError, match="active submission"):
            with capture.execution.submit():
                pass
    with capture.execution.submit():
        with pytest.raises(RuntimeError, match="active submission"):
            with pool.execution.submit():
                pass


def test_completion_keeps_inputs_alive_until_event_finishes():
    import weakref
    from types import SimpleNamespace
    from lightllm.platform import DeviceRuntime

    completed = False
    event = SimpleNamespace(
        record=lambda stream: None,
        query=lambda: completed,
        synchronize=lambda: None,
    )
    runtime = DeviceRuntime("cpu")
    runtime.current_stream = lambda: "stream"
    runtime.create_event = lambda: event
    workspace = AttentionWorkspace(runtime=runtime)
    tensor = torch.ones(2)
    reference = weakref.ref(tensor)
    with workspace.execution.submit(retain=(tensor,)):
        pass
    del tensor
    assert reference() is not None
    with workspace.execution.submit():
        assert reference() is not None
    completed = True
    with workspace.execution.submit():
        assert reference() is None


def test_failed_completion_quarantines_pool_and_preserves_original_error(monkeypatch):
    workspace = AttentionWorkspace("cpu")

    def fail(*args):
        raise RuntimeError("event failure")

    monkeypatch.setattr(workspace.runtime, "current_stream", lambda: "stream")
    monkeypatch.setattr(workspace.runtime, "create_event", fail)
    retained = object()
    with pytest.raises(ValueError, match="original"):
        with workspace.execution.submit(retain=(retained,)):
            raise ValueError("original")
    assert workspace.execution._pending == [(None, [retained])]
    with pytest.raises(RuntimeError, match="quarantined"):
        with workspace.borrow_for_capture().execution.submit():
            pass
    with pytest.raises(RuntimeError, match="quarantined"):
        workspace.execution.synchronize()


def test_output_reader_fences_other_graphs_sharing_the_pool():
    from types import SimpleNamespace
    from lightllm.common.basemodel.graph import CapturedGraph
    from lightllm.platform import DeviceRuntime

    calls = []
    runtime = DeviceRuntime("cpu")
    event = object()
    stream = SimpleNamespace(wait_event=lambda incoming: calls.append(incoming))
    runtime.current_stream = lambda: stream
    runtime.create_event = lambda: SimpleNamespace(record=lambda _: None, query=lambda: True)
    pool = AttentionWorkspace(runtime=runtime)
    first = CapturedGraph(SimpleNamespace(reset=lambda: None), runtime=runtime, execution=pool.execution)
    second = CapturedGraph(
        SimpleNamespace(replay=lambda: calls.append("submit"), reset=lambda: None),
        runtime=runtime,
        execution=pool.execution,
    )
    first.record_output_use(event)
    second.replay()
    assert calls == [event, "submit"]
    first.close()
    second.close()
