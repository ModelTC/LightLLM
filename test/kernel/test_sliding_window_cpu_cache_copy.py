import math

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.sliding_window_cpu_cache_copy import (
    copy_cpu_cache_to_kv_buffer,
    copy_kv_buffer_to_cpu_cache,
    copy_sliding_window_state,
)
from lightllm.common.state_cache_manager import SlidingWindowCacheConfig

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
        window_cpu = _random_bits((slots, *config.get_state_shape()), dtype, generator).pin_memory()
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
            cpu_kv_sliding_state=window_cpu,
            cpu_cache_tensor=cpu_cache,
            tp_rank=rank,
            tp_world_size=tp_world_size,
            big_page_token_num=big_page_tokens,
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
        full_gpu, window_pinned = expected_full.cuda(), expected_window.pin_memory()
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
            cpu_kv_sliding_state=window_pinned,
            cpu_cache_tensor=cpu_cache,
            tp_rank=rank,
            tp_world_size=tp_world_size,
            big_page_token_num=big_page_tokens,
            grid_num=3,
        )
        torch.cuda.synchronize()
        _assert_same_bits(full_gpu, expected_full)
        _assert_same_bits(window_pinned, expected_window)
        _assert_same_bits(cpu_cache.view(cpu_page_num, page_bytes), expected_cache)


def test_empty_copy_is_a_noop():
    indexes = torch.empty(0, dtype=torch.int64, device="cuda")
    kwargs = dict(
        mem_indexes=indexes,
        page_indexes=indexes,
        big_page_buffer_ids=indexes,
        gpu_full_att_kv_state=None,
        cpu_kv_sliding_state=None,
        cpu_cache_tensor=None,
        tp_rank=0,
        tp_world_size=1,
        big_page_token_num=16,
    )
    copy_kv_buffer_to_cpu_cache(page_readies=indexes, **kwargs)
    copy_cpu_cache_to_kv_buffer(**kwargs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layers,window,heads,dim", [(3, 17, 2, 6), (5, 129, 4, 64)])
def test_state_copy_preserves_layer_strides_and_cpu_staging_stream_order(dtype, layers, window, heads, dim):
    generator = torch.Generator().manual_seed(71)
    source = _random_bits((layers, 5 * window + 7, heads, dim), dtype, generator)
    source_gpu = source.cuda()
    source_requests = source_gpu[:, : 5 * window].view(layers, 5, window, heads, dim)
    expected = torch.full((layers, 6 * window + 11, heads, dim), -3, dtype=dtype)
    restored = expected.cuda()
    restored_requests = restored[:, : 6 * window].view(layers, 6, window, heads, dim)
    # Checkpoints are size-first; runtime requests are layer-first views of a
    # larger pool, so copying one request must preserve both layer strides.
    checkpoint_pool = torch.zeros((2, layers, window, heads, dim), dtype=dtype, pin_memory=True)
    checkpoint = checkpoint_pool[1]
    staging = torch.empty_like(checkpoint, pin_memory=True)
    staging.zero_()
    assert checkpoint.is_contiguous() and not source_requests[:, 1].is_contiguous()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Keep preceding GPU work pending: a host-side CPU copy_ would bypass it.
        torch.cuda._sleep(1_000_000)
        for src_req, dst_req in [(1, 0), (3, 4)]:
            copy_sliding_window_state(source_requests[:, src_req], checkpoint)
            copy_sliding_window_state(checkpoint, staging)
            copy_sliding_window_state(staging, restored_requests[:, dst_req])
    stream.synchronize()
    for src_start, dst_start in [(window, 0), (3 * window, 4 * window)]:
        expected[:, dst_start : dst_start + window] = source[:, src_start : src_start + window]
    _assert_same_bits(restored, expected)
    _assert_same_bits(checkpoint, source[:, 3 * window : 4 * window])
    _assert_same_bits(staging, source[:, 3 * window : 4 * window])
    assert torch.count_nonzero(checkpoint_pool[0]).item() == 0
