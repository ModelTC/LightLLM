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
