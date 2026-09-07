from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_cpu_cache_copy import (
    copy_cpu_cache_to_kv_buffer,
    copy_kv_buffer_to_cpu_cache,
)
from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow
from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig, SlidingWindowStateCacheManager
from lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm import context_attention_fwd_gemma4_mm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("window,history_len", [(512, 256), (512, 512), (512, 544), (1024, 1056)])
@pytest.mark.parametrize("q_len", [1, 31])
def test_cpu_window_checkpoint_resumes_attention_exactly(window, history_len, q_len):
    torch.manual_seed(42)
    page_size, head_dim, req_idx = 512, 64, 1
    seq_len = history_len + q_len
    # Logical layer 3 shares layer 2's KV: only physical owners are stored.
    config = SlidingWindowCacheConfig({0: 0, 2: 1, 3: 1}, {1: 0}, window, 1, head_dim, 1, head_dim, torch.bfloat16)
    reference = torch.randn((2, seq_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    full_kv = torch.randn((1, history_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    expected_full_kv = full_kv.clone()
    endpoints = list(range(page_size, history_len + 1, page_size))
    if history_len % page_size:
        endpoints.append(history_len)
    pages = SlidingWindowStateCacheManager(len(endpoints), config)
    for page_id, endpoint in enumerate(endpoints):
        positions = torch.arange(max(0, endpoint - window), endpoint, device="cuda")
        pages.state_cache[page_id, :, positions % window] = reference[:, positions]

    int_tensor = lambda values: torch.tensor(values, device="cuda", dtype=torch.int32)
    mem_indexes = torch.full((len(endpoints) * page_size,), -1, device="cuda", dtype=torch.int32)
    mem_indexes[:history_len] = torch.arange(history_len, device="cuda", dtype=torch.int32)
    page_ids = int_tensor(list(range(len(endpoints))))
    cpu_cache = torch.zeros(
        (len(endpoints), 1, 1, 1, config.get_cpu_cache_big_page_bytes(page_size, 1)),
        dtype=torch.uint8,
        pin_memory=True,
    )
    copy_args = dict(
        mem_indexes=mem_indexes,
        page_indexes=page_ids,
        big_page_buffer_ids=page_ids,
        gpu_full_att_kv_state=full_kv,
        gpu_sliding_state=pages.state_cache,
        cpu_cache_tensor=cpu_cache,
        tp_rank=0,
        tp_world_size=1,
        big_page_token_num=page_size,
        sliding_config=config,
    )
    copy_kv_buffer_to_cpu_cache(page_readies=torch.zeros_like(page_ids, dtype=torch.bool), **copy_args)
    full_kv.fill_(-7)
    pages.state_cache.fill_(-9)
    copy_cpu_cache_to_kv_buffer(**copy_args)
    torch.testing.assert_close(full_kv, expected_full_kv, atol=0, rtol=0)

    manager = object.__new__(ReqManagerForSlidingWindow)
    manager.sliding_config, manager.sliding_window = config, window
    manager.scratch_token_num, manager.scratch_start = q_len, 3 * window
    manager.mem_manager = SimpleNamespace(linear_att_big_page_buffers=pages)
    manager.req_to_sliding_window = torch.zeros(
        (2, manager.scratch_start + q_len, 2, head_dim), device="cuda", dtype=torch.bfloat16
    )
    manager.req_to_sliding_window_indexs = torch.full((3, seq_len), -1, device="cuda", dtype=torch.int32)
    manager.restore_big_page_state(len(endpoints) - 1, SimpleNamespace(req_idx=req_idx))
    manager.req_to_sliding_window[:, manager.scratch_start :] = reference[:, history_len:]
    state = SimpleNamespace(
        input_ids=int_tensor([0] * q_len),
        b_req_idx=int_tensor([req_idx]),
        b_seq_len=int_tensor([seq_len]),
        b_q_seq_len=int_tensor([q_len]),
        b_q_start_loc=int_tensor([0]),
        max_q_seq_len=q_len,
    )
    manager.prepare_sliding_window(state)
    q = torch.randn((q_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    reference_indexes = torch.arange(seq_len, device="cuda", dtype=torch.int32).expand(3, -1)
    image_end = int_tensor([0] * q_len)
    for layer_index in [0, 2, 3]:
        physical_layer = config.get_sliding_layer_index(layer_index)
        actual, expected = torch.empty_like(q), torch.empty_like(q)
        for kv, indexes, output in [
            (manager.req_to_sliding_window[physical_layer], manager.req_to_sliding_window_indexs, actual),
            (reference[physical_layer], reference_indexes, expected),
        ]:
            context_attention_fwd_gemma4_mm(
                q,
                kv[:, :1],
                kv[:, 1:],
                output,
                state.b_req_idx,
                state.b_q_start_loc,
                state.b_seq_len,
                int_tensor([history_len]),
                q_len,
                indexes,
                image_end,
                sliding_window=(window - 1, 0),
            )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
