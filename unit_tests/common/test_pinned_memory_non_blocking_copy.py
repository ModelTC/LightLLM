import pytest
import torch
from torch.profiler import ProfilerActivity, profile


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _profile_request_table_copy(non_blocking: bool):
    token_num = 1024 * 1024
    mem_indexes = torch.arange(token_num, dtype=torch.int32, pin_memory=True)
    req_to_token_indexes = torch.empty((1, token_num), dtype=torch.int32, device="cuda")

    # 先完成分配等准备操作，避免其 CUDA runtime 调用混入待验证区间。
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as prof:
        req_to_token_indexes[0].copy_(mem_indexes, non_blocking=non_blocking)
        if non_blocking:
            # 异步路径需要在读取结果前显式等待；这里会记录为 cudaDeviceSynchronize，
            # 与 copy_ 内部可能产生的 cudaStreamSynchronize 可以明确区分。
            torch.cuda.synchronize()

    runtime_calls = {event.key for event in prof.key_averages()}
    assert torch.equal(req_to_token_indexes[0].cpu(), mem_indexes)
    return runtime_calls


def test_pinned_cpu_to_gpu_non_blocking_copy_avoids_internal_stream_sync():
    blocking_calls = _profile_request_table_copy(non_blocking=False)
    non_blocking_calls = _profile_request_table_copy(non_blocking=True)

    # 两条路径底层都提交异步 H2D memcpy；同步路径随后在 copy_ 内部等待当前 stream。
    assert "cudaMemcpyAsync" in blocking_calls
    assert "cudaMemcpyAsync" in non_blocking_calls
    assert "cudaStreamSynchronize" in blocking_calls
    assert "cudaStreamSynchronize" not in non_blocking_calls
