import pytest
import torch

from lightllm.common.attention.backends._workspace import AttentionWorkspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_workspace_cross_stream_reuse_waits_for_writer():
    workspace = AttentionWorkspace("cuda")
    buffer = workspace.get("shared", 4096, torch.int32)
    buffer.zero_()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(2_000_000)
        buffer.fill_(17)
        workspace.finish()
    with torch.cuda.stream(consumer):
        workspace.begin()
        result = buffer.clone()
        workspace.finish()
    consumer.synchronize()
    assert result.eq(17).all()

    larger = workspace.get("shared", 8192, torch.int32)
    assert larger.data_ptr() != buffer.data_ptr()
    assert buffer.eq(17).all()


def test_shared_object_type_and_frozen_fork_lifetime():
    workspace = AttentionWorkspace("cpu")
    fork = workspace.fork()
    shared = fork.get_object("provider_plan", dict)
    assert workspace.get_object("provider_plan", dict) is shared
    with pytest.raises(TypeError, match="provider_plan requires list"):
        workspace.get_object("provider_plan", list)
    with pytest.raises(TypeError, match="bad requires dict"):
        workspace.get_object("bad", lambda: [], expected_type=dict)
    assert "bad" not in workspace._objects
    buffer = fork.get("scratch", 4)
    fork.freeze()
    workspace.get("scratch", 8)
    assert fork.get("scratch", 4) is buffer
    with pytest.raises(RuntimeError, match="cannot grow"):
        fork.get("scratch", 8)
    with pytest.raises(RuntimeError, match="before capture"):
        fork.get_object("next_bucket", dict)
    added = workspace.get_object("next_bucket", dict)
    assert fork.get_object("next_bucket", dict) is added


def test_cpu_stream_and_event_have_consistent_noop_lifecycle():
    workspace = AttentionWorkspace("cpu")
    runtime = workspace.runtime
    assert runtime.create_stream() is None
    assert runtime.create_event() is None
    with runtime.use_stream(runtime.create_stream()):
        workspace.begin()
        workspace.finish()
