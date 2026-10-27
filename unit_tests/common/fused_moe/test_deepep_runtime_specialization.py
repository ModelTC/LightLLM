import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
    _ep_compact_metadata_kernel,
    _ep_gather_chunk_kernel,
    ep_compact_metadata,
    ep_gather_chunk,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_ep_runtime_counts_reuse_compilation(monkeypatch):
    torch.manual_seed(51)
    hashes = {"gather": set(), "compact": set()}
    for name, kernel in [("gather", _ep_gather_chunk_kernel), ("compact", _ep_compact_metadata_kernel)]:
        original = kernel.run

        def observe(*args, _name=name, _original=original, **kwargs):
            compiled = _original(*args, **kwargs)
            hashes[_name].add(compiled.hash)
            return compiled

        monkeypatch.setattr(kernel, "run", observe)
    for tokens in [1, 3, 16, 17, 33, 129]:
        metadata = torch.randint(-1, 32, (tokens, 10), dtype=torch.int32, device="cuda")
        weights = torch.randn(32, dtype=torch.float32, device="cuda")
        for start, count in [(0, 1), (1, 3), (3, 16)]:
            chunk = torch.randn((count, 129), dtype=torch.bfloat16, device="cuda")
            output = torch.zeros((tokens, 129), dtype=chunk.dtype, device="cuda")
            slots = metadata[:, 2:].long()
            valid = (slots >= start) & (slots < start + count)
            selected = chunk[(slots - start).clamp(0, count - 1)].float()
            reference = (selected * weights[slots.clamp(0, 31), None] * valid[..., None]).sum(1)
            ep_gather_chunk(chunk, start, weights, metadata, output)
            torch.testing.assert_close(output.float(), reference, atol=0.04, rtol=0.01)
        original_header = metadata[:, :2].clone()
        ep_compact_metadata(metadata)
        assert torch.equal(metadata[:, :2], original_header)
        assert torch.equal(metadata[:, 2], torch.arange(tokens, device="cuda", dtype=torch.int32))
        assert (metadata[:, 3:] == -1).all()
    assert all(len(values) == 1 for values in hashes.values()), hashes
