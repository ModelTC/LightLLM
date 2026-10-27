import pytest
import torch

from lightllm.models.glm5_next.triton_kernel import kpool


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def record_kernels(monkeypatch, kernel):
    hashes = []
    original = kernel.run

    def run(*args, **kwargs):
        compiled = original(*args, **kwargs)
        hashes.append(compiled.hash)
        return compiled

    monkeypatch.setattr(kernel, "run", run)
    return hashes


def test_pool_ranges_reuses_binary_across_context_and_chunk_lengths(monkeypatch):
    hashes = record_kernels(monkeypatch, kpool._get_pool_ranges)
    for tokens, max_pools in [
        (5, 7),
        (16, 8),
        (17, 31),
        (255, 32),
        (256, 33),
        (257, 1025),
        (513, 262144),
    ]:
        split = tokens // 2
        lengths = torch.arange(tokens, device="cuda", dtype=torch.int32) % (max_pools * 4) + 1
        cu = torch.tensor([0, split, tokens], device="cuda", dtype=torch.int32)
        expected_starts = torch.cat(
            (
                torch.zeros_like(lengths[:split]),
                torch.full_like(lengths[split:], max_pools),
            )
        )
        expected_lengths = lengths // 4
        actual = kpool.get_pool_ranges(lengths, cu, tokens - split, max_pools)
        for output, expected in zip(
            actual,
            (expected_starts, expected_starts + expected_lengths, expected_lengths),
        ):
            assert torch.equal(output, expected)
    assert len(set(hashes)) == 1


@pytest.mark.parametrize("paged", [False, True])
def test_pool_gather_reuses_binary_across_context_lengths(monkeypatch, paged):
    kernel = kpool._gather_paged_pools if paged else kpool._gather_pools
    hashes = record_kernels(monkeypatch, kernel)
    storage = torch.zeros(64, 1, 584, device="cuda", dtype=torch.bfloat16)
    packed = storage.view(torch.uint8)[:, :, -132:]
    source_keys = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
    source_scales = torch.rand(64, device="cuda") + 0.1
    packed[:, 0, :128] = source_keys
    packed[:, 0, 128:] = source_scales.view(torch.uint8).view(64, 4)
    # A running service keeps its request-table stride fixed as contexts vary.
    table = torch.full((2, 4096 * 4), -1, device="cuda", dtype=torch.int32)
    locations = torch.randint(0, 64, (2, 4096), device="cuda", dtype=torch.int32)
    table[:, 3::4] = locations
    reqs = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    for max_pools in [7, 8, 16, 31, 32, 33, 63, 64, 65, 127, 129, 263, 1025]:
        counts = [max_pools, max_pools - 1]
        lengths = torch.tensor(counts, device="cuda", dtype=torch.int32)
        if paged:
            pages, block_table = kpool.gather_paged_pools(packed, table, reqs, lengths, max_pools)
            page_bytes = pages.view(pages.shape[0], -1)
        else:
            keys, scales = kpool.gather_pools(packed, table, reqs, lengths * 4, max_pools)
            keys = keys.view(torch.uint8).view(2, max_pools, 128)
            scales = scales.view(2, max_pools)
        for row, count in enumerate(counts):
            if paged:
                ids = block_table[row].long()
                actual_keys = page_bytes[ids, : 64 * 128].reshape(-1, 128)[:count]
                actual_scales = page_bytes[ids, 64 * 128 :].contiguous().view(torch.float32).flatten()[:count]
            else:
                actual_keys, actual_scales = keys[row, :count], scales[row, :count]
            locs = locations[1 - row, :count].long()
            assert torch.equal(actual_keys, source_keys[locs])
            assert torch.equal(actual_scales, source_scales[locs])
    assert len(set(hashes)) == 1
