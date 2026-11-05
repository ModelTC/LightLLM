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
