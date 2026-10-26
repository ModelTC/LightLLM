import pytest
import torch

from lightllm.models.glm5_next.triton_kernel import (
    context_flashattention_nopad_fp8 as fp8,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def record_hashes(monkeypatch, kernel):
    hashes = []
    original = kernel.run

    def run(*args, **kwargs):
        compiled = original(*args, **kwargs)
        hashes.append(compiled.hash)
        return compiled

    monkeypatch.setattr(kernel, "run", run)
    return hashes


def test_selected_prefill_compaction_reuses_kernel_across_context_lengths(monkeypatch):
    hashes = record_hashes(monkeypatch, fp8._selected_prefill_compact_kernel)
    topk = torch.full((2, 2176), -1, dtype=torch.int32, device="cuda")
    topk[:, :32] = torch.arange(32, dtype=torch.int32, device="cuda")
    compact = torch.empty((2, 2176), dtype=torch.int64, device="cuda")
    lengths = torch.empty(2, dtype=torch.int32, device="cuda")
    pages = torch.empty((2, 2176), dtype=torch.int32, device="cuda")
    cuq = torch.empty(3, dtype=torch.int32, device="cuda")
    for size in [65536, 65537, 98304, 131071]:
        mapping = torch.arange(size, dtype=torch.int32, device="cuda") % 8192
        starts = torch.tensor([0, size // 4], dtype=torch.int32, device="cuda")
        fp8._selected_prefill_compact_kernel[(2,)](
            topk,
            starts,
            mapping,
            compact,
            lengths,
            pages,
            cuq,
            size,
            8192,
            topk.stride(0),
            topk.stride(1),
            BATCH=2,
            BLOCK=4096,
            num_warps=8,
        )
        torch.testing.assert_close(lengths, torch.full_like(lengths, 32))
        torch.testing.assert_close(
            compact[:, :32], mapping[(starts[:, None] + topk[:, :32]).long()].long()
        )
        torch.testing.assert_close(
            cuq, torch.arange(3, dtype=torch.int32, device="cuda")
        )
    assert len(hashes) == 4 and len(set(hashes)) == 1


def test_online_attention_merge_reuses_kernel_across_query_counts(monkeypatch):
    hashes = record_hashes(monkeypatch, fp8._merge_att_output_kernel)
    for nq in [3, 8, 17, 33, 64]:
        output = torch.randn(nq, 4, 512, dtype=torch.float32, device="cuda")
        lse = torch.randn(4, nq, dtype=torch.float32, device="cuda")
        chunk = torch.randn_like(output)
        chunk_lse = torch.randn_like(lse)
        expected_lse = torch.logaddexp(lse, chunk_lse)
        expected = output * torch.exp(lse - expected_lse).T.unsqueeze(
            -1
        ) + chunk * torch.exp(chunk_lse - expected_lse).T.unsqueeze(-1)
        fp8._merge_att_output_kernel[(nq, 1)](
            output,
            lse,
            chunk,
            chunk_lse,
            chunk_lse.stride(1),
            chunk_lse.stride(0),
            NUM_HEADS=4,
            NUM_Q_TOKENS=nq,
            BLOCK_HEADS=4,
            num_warps=4,
        )
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-6)
    assert len(hashes) == 5 and len(set(hashes)) == 1
