"""Numerical checks require Apple GPU hardware and Torch's shader compiler."""

import pytest
import torch

from lightllm.backends.metal.paged_attention import decode


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires MPS and torch.mps.compile_shader",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("lengths", [(1,), (0, 1, 127, 256), (1, 255, 256, 257), (1, 257, 2048, 8192)])
def test_paged_decode_matches_cpu_with_shared_strided_cache(dtype, dim, lengths):
    """Check partition boundaries, GQA, storage offsets and untouched KV slots."""
    generator = torch.Generator().manual_seed(9381)
    batch, heads, kv_heads, slots = len(lengths), 8, 2, 1024
    maximum = max(lengths)
    host_pool = torch.randn(slots + 1, 2 * kv_heads, dim, dtype=dtype, generator=generator)
    host_query = torch.randn(batch, heads, dim, dtype=dtype, generator=generator) * 3
    host_table = torch.randint(slots, (2 * batch + 1, maximum), generator=generator)
    host_requests = torch.arange(batch) * 2 + 1
    pool = host_pool.to("mps")
    keys, values = pool[1:, :kv_heads], pool[1:, kv_heads:]
    query_storage = torch.empty(batch + 1, heads, 2 * dim + 1, dtype=dtype, device="mps")
    query = query_storage[1:, :, 1::2]
    query.copy_(host_query)
    table_storage = torch.empty(4 * batch + 3, 2 * maximum + 1, dtype=torch.int64, device="mps")
    table = table_storage[1::2, 1::2]
    table.copy_(host_table)
    request_storage = torch.empty(2 * batch + 1, dtype=torch.int32, device="mps")
    requests = request_storage[1::2]
    requests.copy_(host_requests)
    length_storage = torch.empty(2 * batch + 1, dtype=torch.int64, device="mps")
    device_lengths = length_storage[1::2]
    device_lengths.copy_(torch.tensor(lengths))

    with torch.inference_mode():
        output = decode(query, keys, values, table, requests, device_lengths, maximum)
    assert output is not None
    expected = torch.zeros((batch, heads, dim))
    for row, length in enumerate(lengths):
        if length:
            indexes = host_table[host_requests[row], :length]
            key = host_pool[1:, :kv_heads].index_select(0, indexes).float()
            value = host_pool[1:, kv_heads:].index_select(0, indexes).float()
            key = key.repeat_interleave(heads // kv_heads, dim=1)
            value = value.repeat_interleave(heads // kv_heads, dim=1)
            scores = torch.einsum("hd,nhd->hn", host_query[row].float(), key) / dim ** 0.5
            expected[row] = torch.einsum("hn,nhd->hd", scores.softmax(-1), value)

    torch.testing.assert_close(
        output.cpu().float(),
        expected,
        atol=2e-3 if dtype == torch.float16 else 1e-5,
        rtol=2e-3 if dtype == torch.float16 else 3e-5,
    )
    torch.testing.assert_close(pool.cpu(), host_pool, atol=0, rtol=0)


def test_unsupported_decode_selects_fallback():
    query = torch.empty(1, 8, 256, dtype=torch.float16, device="mps")
    keys = torch.empty(16, 2, 256, dtype=query.dtype, device="mps")
    table = torch.zeros(1, 16, dtype=torch.int32, device="mps")
    requests = torch.zeros(1, dtype=torch.int32, device="mps")
    lengths = torch.ones(1, dtype=torch.int32, device="mps")
    assert decode(query.cpu(), keys, keys, table, requests, lengths, 1) is None
    assert decode(query, keys, keys, table, requests, lengths, None) is None
    assert decode(query, keys, keys, table, requests, lengths, 17) is None
    assert decode(query[:, :, :64], keys[:, :, :64], keys[:, :, :64], table, requests, lengths, 1) is None
