"""Byte-level coverage for the prefill indexer gather, including graph reuse."""

import pytest
import torch

from lightllm.models.glm5_next.triton_kernel.kpool import gather_pools


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("pools", [17, 640, 262144])
def test_strided_pools_and_scales_with_empty_and_partial_rows(pools):
    torch.manual_seed(29)
    storage = torch.empty((1024, 1, 584), dtype=torch.bfloat16, device="cuda")
    packed = storage.view(torch.uint8)[:, :, -132:]
    packed[:, 0, :128] = torch.randn(1024, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.pow(2.0, torch.randint(-4, 3, (1024,), device="cuda")).float()
    packed[:, 0, 128:] = scales.view(torch.uint8).view(-1, 4)
    table = torch.full((4, pools * 4), -1, device="cuda", dtype=torch.int32)
    table[:, 3::4] = torch.randint(0, 1024, (4, pools), device="cuda", dtype=torch.int32)
    req = torch.tensor([3, 1, 0, 2], device="cuda", dtype=torch.int32)
    lengths = torch.zeros(4, device="cuda", dtype=torch.int32)
    # Warm up before capture; lengths and slot IDs change on the same graph.
    gather_pools(packed, table, req, lengths, pools)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keys, scales = gather_pools(packed, table, req, lengths, pools)
    for values in [[0, 3, pools * 4 - 3, pools * 4], [pools * 4, 0, 7, 65]]:
        lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32).clamp_max(pools * 4))
        req.copy_(req.roll(1))
        graph.replay()
        positions = torch.arange(pools, device="cuda")
        valid = positions[None, :] < (lengths // 4)[:, None]
        loc = table[req.long(), 3::4].long()
        expected_keys = packed[loc, 0, :128].masked_fill(~valid[:, :, None], 0)
        expected_scales = packed[:, 0, 128:].contiguous().view(torch.float32)[loc].view(4, pools)
        expected_scales.masked_fill_(~valid, 1.0)
        torch.testing.assert_close(keys.view(torch.uint8).view(4, pools, 128), expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(scales.view(4, pools), expected_scales, rtol=0, atol=0)
