import math

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_cpu_cache_copy import (
    copy_cpu_cache_to_kv_buffer,
    copy_kv_buffer_to_cpu_cache,
)
from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _random_bits(shape, dtype, generator):
    data = torch.randint(
        256,
        (math.prod(shape) * dtype.itemsize,),
        dtype=torch.uint8,
        generator=generator,
    )
    return data.view(dtype).reshape(shape)


def _assert_same_bits(actual, expected):
    torch.testing.assert_close(actual.cpu().view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)


@pytest.mark.parametrize("tp_world_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout",
    [
        (2, 3, 7, 1, 64, 2, 32),
        (4, 1, 5, 2, 32, 1, 128),
        # Both full KV and window state span multiple blocks per program.
        (9, 2, 257, 4, 256, 1, 128),
        # Deliberately leaves alignment padding in the CPU page for TP=1.
        (1, 1, 2, 1, 6, 1, 6),
    ],
)
def test_multi_page_round_trip_preserves_tp_slices_tail_and_ring(tp_world_size, dtype, layout):
    (
        full_layers,
        sliding_layers,
        window,
        full_heads,
        full_dim,
        sliding_heads,
        sliding_dim,
    ) = layout
    big_page_tokens, page_num, cpu_page_num, slots = 5, 4, 5, 5
    token_num = page_num * big_page_tokens + 7
    # Duplicate logical layers model shared read-only owners. Only distinct
    # physical owners must be copied into the CPU page.
    sliding_map = {index: index for index in range(sliding_layers)}
    sliding_map[sliding_layers + full_layers] = sliding_layers - 1
    full_map = {sliding_layers + index: index for index in range(full_layers)}
    config = SlidingWindowCacheConfig(
        sliding_map,
        full_map,
        window,
        sliding_heads,
        sliding_dim,
        full_heads,
        full_dim,
        dtype,
    )
    page_bytes = config.get_cpu_cache_big_page_bytes(big_page_tokens, tp_world_size)
    full_rank_bytes = config.get_cpu_cache_full_att_bytes(big_page_tokens, tp_world_size) // tp_world_size
    window_rank_bytes = config.get_cpu_cache_state_bytes(tp_world_size) // tp_world_size
    full_total_bytes = full_rank_bytes * tp_world_size
    token_bytes = full_rank_bytes // big_page_tokens
    cpu_cache = torch.full((cpu_page_num, 1, 1, 1, page_bytes), 0xAB, dtype=torch.uint8, pin_memory=True)
    expected_cache = cpu_cache.view(cpu_page_num, page_bytes).clone()
    generator = torch.Generator().manual_seed(47)

    mem_indexes = torch.randperm(token_num, generator=generator)[: page_num * big_page_tokens].reshape(page_num, -1)
    mem_indexes[-1, -2:] = -1
    page_indexes = torch.tensor([3, 0, -1, 2], dtype=torch.int32)
    page_readies = torch.tensor([False, True, False, False])
    # A skipped page must not read its checkpoint ID or dereference slot -1.
    big_page_ids = torch.tensor([4, -1, -1, 2], dtype=torch.int64)
    sources = []
    for rank in range(tp_world_size):
        full_cpu = _random_bits((full_layers, token_num, 2 * full_heads, full_dim), dtype, generator)
        window_cpu = _random_bits((slots, *config.get_state_shape()), dtype, generator)
        sources.append((full_cpu, window_cpu))
        for page in [0, 3]:
            cpu_page = page_indexes[page].item()
            for offset, token in enumerate(mem_indexes[page].tolist()):
                if token != -1:
                    start = rank * full_rank_bytes + offset * token_bytes
                    expected_cache[cpu_page, start : start + token_bytes].copy_(
                        full_cpu[:, token].contiguous().view(torch.uint8).flatten()
                    )
            start = full_total_bytes + rank * window_rank_bytes
            expected_cache[cpu_page, start : start + window_rank_bytes].copy_(
                window_cpu[big_page_ids[page]].view(torch.uint8).flatten()
            )
        copy_kv_buffer_to_cpu_cache(
            mem_indexes=mem_indexes.flatten().cuda(),
            page_indexes=page_indexes.cuda(),
            page_readies=page_readies.cuda(),
            big_page_buffer_ids=big_page_ids.cuda(),
            gpu_full_att_kv_state=full_cpu.cuda(),
            gpu_sliding_state=window_cpu.cuda(),
            cpu_cache_tensor=cpu_cache,
            tp_rank=rank,
            tp_world_size=tp_world_size,
            big_page_token_num=big_page_tokens,
            sliding_config=config,
            grid_num=3,
        )
        torch.cuda.synchronize()
        # This also checks ready/invalid pages, invalid tail tokens, other TP
        # ranks, untouched CPU pages and final alignment padding.
        _assert_same_bits(cpu_cache.view(cpu_page_num, page_bytes), expected_cache)

    load_indexes = torch.randperm(token_num, generator=generator)[: page_num * big_page_tokens].reshape(page_num, -1)
    load_indexes[-1, -2:] = -1
    load_pages = torch.tensor([3, -1, -1, 2], dtype=torch.int32, device="cuda")
    load_slots = torch.tensor([1, -1, -1, 4], dtype=torch.int64)
    for rank, (full_cpu, window_cpu) in enumerate(sources):
        expected_full = torch.full_like(full_cpu.view(torch.uint8), 0xCD).view(dtype)
        expected_window = torch.full_like(window_cpu.view(torch.uint8), 0xCD).view(dtype)
        full_gpu, window_gpu = expected_full.cuda(), expected_window.cuda()
        for page in [0, 3]:
            for offset, target in enumerate(load_indexes[page].tolist()):
                if target != -1:
                    expected_full[:, target].copy_(full_cpu[:, mem_indexes[page, offset]])
            # The ring's physical order must be unchanged, including when the
            # token page length differs from the sliding window length.
            expected_window[load_slots[page]].copy_(window_cpu[big_page_ids[page]])
        copy_cpu_cache_to_kv_buffer(
            mem_indexes=load_indexes.flatten().cuda(),
            page_indexes=load_pages,
            big_page_buffer_ids=load_slots.cuda(),
            gpu_full_att_kv_state=full_gpu,
            gpu_sliding_state=window_gpu,
            cpu_cache_tensor=cpu_cache,
            tp_rank=rank,
            tp_world_size=tp_world_size,
            big_page_token_num=big_page_tokens,
            sliding_config=config,
            grid_num=3,
        )
        torch.cuda.synchronize()
        _assert_same_bits(full_gpu, expected_full)
        _assert_same_bits(window_gpu, expected_window)
        _assert_same_bits(cpu_cache.view(cpu_page_num, page_bytes), expected_cache)


def test_empty_copy_is_a_noop():
    config = SlidingWindowCacheConfig({0: 0}, {1: 0}, 8, 1, 64, 1, 64, torch.bfloat16)
    indexes = torch.empty(0, dtype=torch.int64, device="cuda")
    kwargs = dict(
        mem_indexes=indexes,
        page_indexes=indexes,
        big_page_buffer_ids=indexes,
        gpu_full_att_kv_state=None,
        gpu_sliding_state=None,
        cpu_cache_tensor=None,
        tp_rank=0,
        tp_world_size=1,
        big_page_token_num=16,
        sliding_config=config,
    )
    copy_kv_buffer_to_cpu_cache(page_readies=indexes, **kwargs)
    copy_cpu_cache_to_kv_buffer(**kwargs)
