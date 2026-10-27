import pytest
import torch
from triton.runtime.jit import JITFunction

from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import _compact_topk_indices_kernel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("first_mode", [True, False])
def test_compact_modes_preserve_layout_and_reuse_each_variant(monkeypatch, first_mode):
    torch.manual_seed(1525)
    misses, hashes = [], {}
    original = JITFunction._do_compile

    def observe(self, *args, **kwargs):
        if self is _compact_topk_indices_kernel:
            misses.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "_do_compile", observe)
    for index, mode in enumerate([first_mode, not first_mode, first_mode, not first_mode]):
        batch = [1, 3, 33, 65][index]
        topk = torch.randint(-7, 20000, (batch, 2176), device="cuda", dtype=torch.int32)
        topk[:, ::17] = -1
        starts = torch.arange(batch, device="cuda", dtype=torch.int32) * 31
        output = torch.full_like(topk, -99999)
        lengths = torch.full((batch,), -987, device="cuda", dtype=torch.int32)
        low, high = [(0, 1), (127, 2305), (8193, 16385), (0, 30000)][index]
        before_misses = len(misses)
        compiled = _compact_topk_indices_kernel[(batch,)](
            topk,
            starts,
            output,
            lengths,
            low,
            high,
            TOPK=2176,
            BLOCK=4096,
            COMPACT=mode,
        )
        if mode in hashes:
            assert len(misses) == before_misses
        hashes.setdefault(mode, set()).add(compiled.hash)
        ragged = topk + starts[:, None]
        valid = (topk >= 0) & (ragged >= low) & (ragged < high)
        if mode:
            assert torch.equal(lengths, valid.sum(1).int())
            for row in range(batch):
                expected = ragged[row][valid[row]] - low
                assert torch.equal(output[row, : expected.numel()], expected)
                assert (output[row, expected.numel() :] == -99999).all()
        else:
            expected = torch.where(valid, ragged - low, -1)
            assert torch.equal(output, expected)
            assert (lengths == -987).all()
    # Keep separate constexpr modes: measurements found the specialized
    # compact/noncompact variants faster than a uniform runtime branch.
    assert set(hashes) == {True, False}
    assert all(len(variants) == 1 for variants in hashes.values())
    assert len(set.union(*hashes.values())) == 2
