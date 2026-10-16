from types import SimpleNamespace

import numpy as np
import pytest
import torch

# 沿用生产导入顺序，避免 basemodel 与 req_manager 的模块级循环导入。
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.req_manager import req_sampling_params
from lightllm.common.req_manager.req_sampling_params import ReqSamplingParamsManager
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import token_id_counter


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_prompt_ids_gpu_buffer_reuses_per_stream_and_isolates_async_copies(monkeypatch):
    monkeypatch.setattr(req_sampling_params, "get_env_start_args", lambda: SimpleNamespace(max_req_total_len=32))
    manager = ReqSamplingParamsManager.__new__(ReqSamplingParamsManager)
    stream0, stream1 = torch.cuda.Stream(), torch.cuda.Stream()
    source0 = torch.full((96,), 11, dtype=torch.int32, pin_memory=True)
    source1 = torch.full((96,), 22, dtype=torch.int32, pin_memory=True)
    assert not hasattr(manager, "_prompt_ids_gpu_buffers")
    with torch.cuda.stream(stream0):
        first = manager._get_prompt_ids_gpu_buffer()
        assert manager._get_prompt_ids_gpu_buffer() is first
        assert first.shape == (96,)
        assert first.dtype == torch.int32 and first.device == stream0.device
        torch.cuda._sleep(100_000_000)
        first.copy_(source0, non_blocking=True)
        completed0 = torch.cuda.Event()
        completed0.record()
    with torch.cuda.stream(stream1):
        other = manager._get_prompt_ids_gpu_buffer()
        assert manager._get_prompt_ids_gpu_buffer() is other
        assert other.data_ptr() != first.data_ptr()
        assert other.shape == (96,)
        torch.cuda._sleep(100_000_000)
        other.copy_(source1, non_blocking=True)
        completed1 = torch.cuda.Event()
        completed1.record()
    with torch.cuda.stream(stream0):
        assert manager._get_prompt_ids_gpu_buffer() is first
    completed0.synchronize()
    completed1.synchronize()
    assert torch.equal(first.cpu(), source0)
    assert torch.equal(other.cpu(), source1)
    assert len(manager._prompt_ids_gpu_buffers) == 2


@pytest.mark.parametrize("mode", ["gpu_counter", "pin_mem_counter"])
def test_prompt_counter_initialization_reuses_stream_buffer(monkeypatch, mode):
    monkeypatch.setattr(req_sampling_params, "get_env_start_args", lambda: SimpleNamespace(max_req_total_len=8192))
    manager = ReqSamplingParamsManager.__new__(ReqSamplingParamsManager)
    manager.penalty_counter_mode = mode
    manager.req_to_next_token_ids = torch.zeros((8, 1), dtype=torch.int64, device="cuda")
    manager.req_to_next_token_scores = None
    for name in (
        "req_to_presence_penalty",
        "req_to_frequency_penalty",
        "req_to_repetition_penalty",
        "req_to_exponential_decay_length_penalty",
    ):
        setattr(manager, name, torch.zeros(8, device="cuda"))
    manager.req_to_out_token_id_counter = torch.zeros(
        (8, 16),
        dtype=torch.int32,
        device="cuda" if mode == "gpu_counter" else "cpu",
        pin_memory=mode == "pin_mem_counter",
    )
    # 提前编译计数 kernel，随后延迟 GPU，检验连续复用暂存区时的异步执行顺序。
    token_id_counter(torch.zeros(1, dtype=torch.int32, device="cuda"), manager.req_to_out_token_id_counter[0])
    torch.cuda.synchronize()

    def forbid_synchronize(*args, **kwargs):
        pytest.fail("request initialization must not synchronize its CUDA stream")

    monkeypatch.setattr(torch.cuda.Stream, "synchronize", forbid_synchronize)
    stream = torch.cuda.Stream()
    buffer_ptrs = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        for index in range(8):
            params = SimpleNamespace(
                presence_penalty=1.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                input_penalty=True,
                exponential_decay_length_penalty=SimpleNamespace(to_tuple=lambda: (0, 1.0)),
            )
            prompt = np.full(4096, index + 1, dtype=np.int64)
            req = SimpleNamespace(
                req_idx=index,
                sampling_param=SimpleNamespace(shm_param=params),
                shm_req=SimpleNamespace(
                    alloc_shm_numpy_len=8192,
                    get_prompt_ids_numpy=lambda prompt=prompt: prompt,
                ),
                get_last_gen_token=lambda: 0,
            )
            manager.init_req_sampling_params(req)
            buffer_ptrs.append(manager._get_prompt_ids_gpu_buffer().data_ptr())
        completed = torch.cuda.Event()
        completed.record()
    completed.synchronize()
    expected = torch.zeros((8, 16), dtype=torch.int32)
    expected[torch.arange(8), torch.arange(1, 9)] = 4096
    assert torch.equal(manager.req_to_out_token_id_counter.cpu(), expected)
    assert len(set(buffer_ptrs)) == len(manager._prompt_ids_gpu_buffers) == 1
