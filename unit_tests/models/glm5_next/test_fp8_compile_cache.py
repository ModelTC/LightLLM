import pytest
import torch
import triton

from lightllm.models.glm5_next.triton_kernel import (
    context_flashattention_nopad_fp8 as fp8,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def record_kernel_hashes(monkeypatch, kernel):
    hashes = []
    original = kernel.run

    def run(*args, **kwargs):
        compiled = original(*args, **kwargs)
        hashes.append(compiled.hash)
        return compiled

    monkeypatch.setattr(kernel, "run", run)
    return hashes


def test_selected_prefill_compaction_reuses_kernel_across_context_lengths(monkeypatch):
    hashes = record_kernel_hashes(monkeypatch, fp8._selected_prefill_compact_kernel)
    topk = torch.full((2, 2176), -1, dtype=torch.int32, device="cuda")
    topk[:, :32] = torch.arange(32, dtype=torch.int32, device="cuda")
    compact = torch.empty((2, 2176), dtype=torch.int64, device="cuda")
    lengths = torch.empty(2, dtype=torch.int32, device="cuda")
    pages = torch.empty((2, 2176), dtype=torch.int32, device="cuda")
    cu_q = torch.empty(3, dtype=torch.int32, device="cuda")
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
            cu_q,
            size,
            8192,
            topk.stride(0),
            topk.stride(1),
            BATCH=2,
            BLOCK=4096,
            num_warps=8,
        )
        torch.testing.assert_close(lengths, torch.full_like(lengths, 32))
        torch.testing.assert_close(compact[:, :32], mapping[(starts[:, None] + topk[:, :32]).long()].long())
        torch.testing.assert_close(cu_q, torch.arange(3, dtype=torch.int32, device="cuda"))
    assert len(hashes) == 4 and len(set(hashes)) == 1


def test_online_attention_merge_reuses_kernel_across_query_counts(monkeypatch):
    hashes = record_kernel_hashes(monkeypatch, fp8._merge_att_output_kernel)
    for num_q_tokens in [3, 8, 17, 33, 64]:
        output = torch.randn(num_q_tokens, 4, 512, dtype=torch.float32, device="cuda")
        lse = torch.randn(4, num_q_tokens, dtype=torch.float32, device="cuda")
        chunk_output = torch.randn_like(output)
        chunk_lse = torch.randn_like(lse)
        expected_lse = torch.logaddexp(lse, chunk_lse)
        expected = output * torch.exp(lse - expected_lse).T.unsqueeze(-1) + chunk_output * torch.exp(
            chunk_lse - expected_lse
        ).T.unsqueeze(-1)
        fp8._merge_att_output_kernel[(num_q_tokens, 1)](
            output,
            lse,
            chunk_output,
            chunk_lse,
            chunk_lse.stride(1),
            chunk_lse.stride(0),
            NUM_HEADS=4,
            NUM_Q_TOKENS=num_q_tokens,
            BLOCK_HEADS=4,
            num_warps=4,
        )
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-6)
    assert len(hashes) == 5 and len(set(hashes)) == 1


def test_prefill_row_table_and_kv_gather_reuse_kernels_across_token_counts(monkeypatch):
    row_table_hashes = record_kernel_hashes(monkeypatch, fp8._build_prefill_row_table_kernel)
    gather_hashes = record_kernel_hashes(monkeypatch, fp8._gather_prefill_kv_kernel)
    cached_kv = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
    packed_kv = torch.zeros(64, 1, 584, dtype=torch.uint8, device="cuda")
    packed_kv[:, 0, :448] = cached_kv[:, :448].to(torch.float8_e4m3fn).view(torch.uint8)
    packed_kv[:, 0, 448:576] = cached_kv[:, 448:].contiguous().view(torch.uint8)
    packed_kv[:, 0, 576:583] = 127
    cached_kv[:, :448] = cached_kv[:, :448].to(torch.float8_e4m3fn).to(torch.bfloat16)
    for num_tokens in [3, 8, 17, 33]:
        prefill_mem_index = torch.arange(0, num_tokens, 2, dtype=torch.int32, device="cuda")
        prefill_row_table = torch.full((64,), -1, dtype=torch.int32, device="cuda")
        prefill_kv = torch.randn(prefill_mem_index.numel(), 1, 512, dtype=torch.bfloat16, device="cuda")
        fp8._build_prefill_row_table_kernel[(1,)](
            prefill_mem_index, prefill_row_table, prefill_mem_index.numel(), BLOCK=256
        )
        mem_index = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
        output = torch.empty(num_tokens, 1, 512, dtype=torch.bfloat16, device="cuda")
        fp8._gather_prefill_kv_kernel[(triton.cdiv(num_tokens, 4),)](
            packed_kv,
            mem_index,
            prefill_row_table,
            prefill_kv,
            output,
            num_tokens,
            packed_kv.stride(0),
            prefill_kv.stride(0),
            prefill_kv.stride(2),
            BLOCK_ROWS=4,
        )
        expected = cached_kv[:num_tokens].clone()
        expected[prefill_mem_index.long()] = prefill_kv[:, 0]
        torch.testing.assert_close(output[:, 0], expected, rtol=0, atol=0)
    assert len(set(row_table_hashes)) == len(set(gather_hashes)) == 1


@pytest.mark.parametrize("compact", [False, True])
def test_topk_compaction_reuses_kernel_across_chunk_boundaries(monkeypatch, compact):
    hashes = record_kernel_hashes(monkeypatch, fp8._compact_topk_indices_kernel)
    topk_indices = torch.tensor([[-1, 0, 2, 3, 8, 17, 32]] * 2, dtype=torch.int32, device="cuda")
    kv_starts = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    ragged_indices = topk_indices + kv_starts[:, None]
    for chunk_start, chunk_end in [(0, 3), (3, 19), (19, 128)]:
        chunk_indices = torch.full_like(topk_indices, -2)
        chunk_lengths = torch.empty(2, dtype=torch.int32, device="cuda")
        fp8._compact_topk_indices_kernel[(2,)](
            topk_indices,
            kv_starts,
            chunk_indices,
            chunk_lengths,
            chunk_start,
            chunk_end,
            TOPK=7,
            BLOCK=8,
            COMPACT=compact,
        )
        valid = (topk_indices >= 0) & (ragged_indices >= chunk_start) & (ragged_indices < chunk_end)
        expected = ragged_indices - chunk_start
        if compact:
            torch.testing.assert_close(chunk_lengths, valid.sum(dim=1).int())
            for row in range(2):
                selected = expected[row, valid[row]]
                torch.testing.assert_close(chunk_indices[row, : selected.numel()], selected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(chunk_indices, expected.masked_fill(~valid, -1), rtol=0, atol=0)
    assert len(set(hashes)) == 1
