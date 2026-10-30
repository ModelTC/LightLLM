"""Byte-exact gather and graph-replay coverage for GLM K-pool staging."""

import pytest
import torch
import triton

from lightllm.models.glm5_next.triton_kernel.kpool import gather_paged_pools, gather_pools


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _inputs(max_pools, requests=8):
    # The packed view has both a nonzero byte offset and a larger row stride.
    storage = torch.empty((128, 1, 584), dtype=torch.bfloat16, device="cuda")
    packed = storage.view(torch.uint8)[:, :, -132:]
    keys = torch.randn(128, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.pow(2.0, torch.randint(-4, 3, (128,), device="cuda")).float()
    packed[:, 0, :128] = keys
    packed[:, 0, 128:] = scales.view(torch.uint8).view(128, 4)
    table_storage = torch.full((requests, max_pools * 4 + 16), -1, device="cuda", dtype=torch.int32)
    table = table_storage[:, 4:-12]
    locations = torch.randint(0, 128, (requests, max_pools), device="cuda", dtype=torch.int32)
    table[:, 3::4] = locations
    return packed, table, locations, keys, scales


@pytest.mark.parametrize("pools", [17, 33, 640, 65536, 262144])
def test_gather_pools_bytes_scales_and_changing_graph_inputs(pools):
    torch.manual_seed(29)
    packed, table, locations, source_keys, source_scales = _inputs(pools)
    req = torch.tensor([3, 1, 0, 2], device="cuda", dtype=torch.int32)
    lengths = torch.zeros(4, device="cuda", dtype=torch.int32)
    gather_pools(packed, table, req, lengths, pools)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keys, scales = gather_pools(packed, table, req, lengths, pools)
    for values in ([0, 3, pools * 4 - 3, pools * 4], [pools * 4, 0, 7, 65], [2, 2, 2, 2]):
        lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32).clamp_max(pools * 4))
        req.copy_(req.roll(1))
        graph.replay()
        valid = torch.arange(pools, device="cuda")[None, :] < (lengths // 4)[:, None]
        loc = locations[req.long()].long()
        expected_keys = source_keys[loc].masked_fill(~valid[:, :, None], 0)
        expected_scales = source_scales[loc].masked_fill(~valid, 1.0)
        torch.testing.assert_close(keys.view(torch.uint8).view(4, pools, 128), expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(scales.view(4, pools), expected_scales, rtol=0, atol=0)


def _check_paged(pages, block_table, req, lengths, locations, source_keys, source_scales):
    page_bytes = pages.view(pages.shape[0], -1)
    for row, (slot, count) in enumerate(zip(req.tolist(), lengths.tolist())):
        # Shared pages may contain the owner's later keys. Only the original
        # per-query causal length is readable by the indexer.
        page_ids = block_table[row, : triton.cdiv(count, 64)].long()
        actual_keys = page_bytes[page_ids, : 64 * 128].reshape(-1, 128)[:count]
        actual_scales = page_bytes[page_ids, 64 * 128 :].contiguous().view(torch.float32).flatten()[:count]
        loc = locations[slot, :count].long()
        torch.testing.assert_close(actual_keys, source_keys[loc], rtol=0, atol=0)
        torch.testing.assert_close(actual_scales, source_scales[loc], rtol=0, atol=0)


@pytest.mark.parametrize("max_pools", [640, 4096, 262144])
@pytest.mark.parametrize("reuse_group", [1, 2, 4, 8])
def test_paged_reuse_mixed_requests_causal_lengths_and_graph_slot_recycling(max_pools, reuse_group):
    torch.manual_seed(30)
    packed, table, locations, source_keys, source_scales = _inputs(max_pools)
    batch = reuse_group * 2 + 1
    shared = [1] * reuse_group + [2 + i % 2 for i in range(reuse_group)] + [4]
    mixed = [1 + i % 3 for i in range(batch)]
    req = torch.tensor(shared, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([max(0, max_pools - i * 63) for i in range(batch)], device="cuda", dtype=torch.int32)
    gather_paged_pools(packed, table, req, lengths, max_pools, reuse_group=reuse_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pages, block_table = gather_paged_pools(packed, table, req, lengths, max_pools, reuse_group=reuse_group)
    identity = torch.arange(block_table.numel(), device="cuda", dtype=torch.int32).view_as(block_table)
    scenarios = [
        (shared, [max(0, max_pools - i * 63) for i in range(batch)]),
        (mixed, [max_pools if i % 2 else min(65, max_pools) for i in range(batch)]),
        (shared, [min(i * 17, max_pools) for i in range(batch)]),
        ([0] * batch, [0] * batch),
        ([5] * reuse_group + mixed[reuse_group:], [max_pools] * batch),
    ]
    for slots, counts in scenarios:
        req.copy_(torch.tensor(slots, device="cuda", dtype=torch.int32))
        lengths.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        # Recycle physical KV locations without changing any captured pointer.
        locations.copy_(locations.roll(1, dims=1))
        table[:, 3::4] = locations
        original_lengths = lengths.clone()
        if not any(counts):
            pages.fill_(127)
        graph.replay()
        torch.testing.assert_close(lengths, original_lengths, rtol=0, atol=0)
        _check_paged(pages, block_table, req, lengths, locations, source_keys, source_scales)
        if max(counts) < 4096 or reuse_group == 1:
            torch.testing.assert_close(block_table, identity, rtol=0, atol=0)
        elif slots == shared and reuse_group > 1:
            valid_pages = triton.cdiv(max(counts[:reuse_group]), 64)
            for row in range(1, reuse_group):
                torch.testing.assert_close(block_table[row, :valid_pages], block_table[0, :valid_pages])
        if not any(counts):
            dummy = pages.view(pages.shape[0], -1)[block_table[:, 0].long()]
            assert not dummy[:, : 64 * 128].any()
            assert (dummy[:, 64 * 128 :].contiguous().view(torch.float32) == 1).all()


@pytest.mark.parametrize("batch", [1, 3, 5])
def test_paged_partial_groups_keep_valid_per_query_pages(batch):
    packed, table, locations, source_keys, source_scales = _inputs(4097)
    req = torch.ones(batch, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([4097 - i for i in range(batch)], device="cuda", dtype=torch.int32)
    pages, block_table = gather_paged_pools(packed, table, req, lengths, 4097, reuse_group=4)
    _check_paged(pages, block_table, req, lengths, locations, source_keys, source_scales)
    if batch < 4:
        expected = torch.arange(block_table.numel(), device="cuda", dtype=torch.int32).view_as(block_table)
        torch.testing.assert_close(block_table, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reuse_group", [0, 3, 16])
def test_reject_unsupported_reuse_group_before_allocating(reuse_group):
    with pytest.raises(ValueError, match="reuse_group"):
        gather_paged_pools(None, None, None, None, 4096, reuse_group=reuse_group)
