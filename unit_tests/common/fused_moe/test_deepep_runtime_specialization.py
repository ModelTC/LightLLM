import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
    _ep_compact_metadata_kernel,
    _ep_gather_chunk_kernel,
    ep_compact_metadata,
    ep_gather_chunk,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("aligned_chunks", [False, True])
def test_ep_runtime_counts_reuse_compilation(monkeypatch, aligned_chunks):
    torch.manual_seed(51)
    # Production chunks are 128-aligned. Generic unaligned calls remain valid,
    # but need not share their boundary specialization with other chunks.
    chunks = [(0, 128), (128, 128), (32768, 128)] if aligned_chunks else [(0, 1), (1, 3), (3, 16)]
    weight_count = max(start + count for start, count in chunks) + 1
    hashes = {"gather": {}, "compact": set()}
    bounds = None
    for name, kernel in [("gather", _ep_gather_chunk_kernel), ("compact", _ep_compact_metadata_kernel)]:
        original = kernel.run

        def observe(*args, _name=name, _original=original, **kwargs):
            compiled = _original(*args, **kwargs)
            if _name == "gather":
                hashes[_name].setdefault(bounds, set()).add(compiled.hash)
            else:
                hashes[_name].add(compiled.hash)
            return compiled

        monkeypatch.setattr(kernel, "run", observe)
    for tokens in [1, 3, 16, 17, 33, 129]:
        metadata = torch.randint(-1, 32, (tokens, 10), dtype=torch.int32, device="cuda")
        weights = torch.randn(weight_count, dtype=torch.float32, device="cuda")
        for start, count in chunks:
            bounds = (start, count)
            metadata[:, 2:] = torch.randint(start, start + count, (tokens, 8), device="cuda", dtype=torch.int32)
            metadata[:, 2] = start - 1
            metadata[:, 3] = start
            metadata[:, 4] = start + count - 1
            metadata[:, 5] = start + count
            chunk = torch.randn((count, 129), dtype=torch.bfloat16, device="cuda")
            output = torch.zeros((tokens, 129), dtype=chunk.dtype, device="cuda")
            slots = metadata[:, 2:].long()
            valid = (slots >= start) & (slots < start + count)
            selected = chunk[(slots - start).clamp(0, count - 1)].float()
            reference = (selected * weights[slots.clamp(0, weight_count - 1), None] * valid[..., None]).sum(1)
            ep_gather_chunk(chunk, start, weights, metadata, output)
            torch.testing.assert_close(output.float(), reference, atol=0.04, rtol=0.01)
        original_header = metadata[:, :2].clone()
        ep_compact_metadata(metadata)
        assert torch.equal(metadata[:, :2], original_header)
        assert torch.equal(metadata[:, 2], torch.arange(tokens, device="cuda", dtype=torch.int32))
        assert (metadata[:, 3:] == -1).all()
    assert len(hashes["compact"]) == 1
    assert all(len(values) == 1 for values in hashes["gather"].values()), hashes
    if aligned_chunks:
        assert len(set.union(*hashes["gather"].values())) == 1


@pytest.mark.parametrize("hidden", [8, 4096], ids=["masked", "unmasked"])
def test_ep_prefill_gather_offsets_above_int32_range(hidden):
    if torch.cuda.mem_get_info()[0] < 32 * 1024 ** 3:
        pytest.skip("Large-stride regression requires 32 GiB of available GPU memory")
    row_stride = 2 ** 30
    output = torch.empty_strided((3, hidden), (row_stride, 1), device="cuda", dtype=torch.bfloat16)
    metadata = torch.empty_strided((3, 4), (row_stride, 1), device="cuda", dtype=torch.int32)
    metadata.copy_(torch.tensor([[0, 0, 0, 2], [1, 0, 1, 3], [2, 0, 2, 3]], device="cuda", dtype=torch.int32))
    weights = torch.tensor([0.5, 0.25, 0.125, 0.0625], device="cuda")
    chunk = torch.tensor([[1] * hidden, [2] * hidden, [4] * hidden, [8] * hidden], device="cuda", dtype=output.dtype)
    ep_gather_chunk(chunk[:2], 0, weights, metadata, output, init_output=True)
    expected = torch.tensor([[0.5] * hidden, [0.5] * hidden, [0] * hidden], device="cuda", dtype=output.dtype)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    ep_gather_chunk(chunk[2:], 2, weights, metadata, output)
    torch.testing.assert_close(output, torch.ones_like(expected), atol=0, rtol=0)
