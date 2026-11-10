import pytest
import torch
from triton.runtime.jit import JITFunction

from lightllm.models.glm5_next.triton_kernel.fp8_prefill import _compact_topk_indices_kernel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("aligned_chunks", [False, True])
def test_compact_topk_preserves_layout_and_reuses_kernel(monkeypatch, index_dtype, aligned_chunks):
    torch.manual_seed(1525)
    misses, hashes = [], set()
    original = JITFunction._do_compile

    def observe(self, *args, **kwargs):
        if self is _compact_topk_indices_kernel:
            misses.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "_do_compile", observe)
    # Exercise aligned production chunks and unaligned chunk boundaries separately.
    chunk_starts = [0, 2097152, 4194304, 0] if aligned_chunks else [127, 8193, 8193, 127]
    for index in range(4):
        batch = [1, 3, 33, 65][index]
        low = chunk_starts[index]
        high = low + [1, 2305, 16385, 30000][index]
        topk = torch.randint(-7, 20000, (batch, 2176), device="cuda", dtype=index_dtype)
        topk[:, ::17] = -1
        base = max(0, low - 32)
        starts = torch.arange(batch, device="cuda", dtype=torch.int32) * 31 + base
        # Include positive indices outside either end of the current chunk.
        topk[0, 1:6] = torch.tensor(
            [low - 2 - base, low - 1 - base, low - base, high - 1 - base, high - base],
            device="cuda",
            dtype=torch.int32,
        )
        output = torch.full_like(topk, -99999)
        lengths = torch.full((batch,), -987, device="cuda", dtype=torch.int32)
        before_misses = len(misses)
        compiled = _compact_topk_indices_kernel[(batch,)](
            topk,
            starts,
            output,
            lengths,
            low,
            high,
            topk.stride(0),
            topk.stride(1),
            TOPK=2176,
            BLOCK=4096,
        )
        if hashes:
            assert len(misses) == before_misses
        hashes.add(compiled.hash)
        ragged = topk + starts[:, None]
        valid = (topk >= 0) & (ragged >= low) & (ragged < high)
        assert torch.equal(lengths, valid.sum(1).int())
        for row in range(batch):
            expected = ragged[row][valid[row]] - low
            assert torch.equal(output[row, : expected.numel()], expected)
            assert (output[row, expected.numel() :] == -99999).all()
    assert len(hashes) == 1
