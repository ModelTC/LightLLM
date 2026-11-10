import pytest
import torch

from lightllm.models.glm5_next.triton_kernel import (
    fp8_prefill as fp8,
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


@pytest.mark.parametrize("transposed_lse", [False, True])
def test_online_attention_merge_reuses_kernel_across_query_counts(monkeypatch, transposed_lse):
    hashes = record_kernel_hashes(monkeypatch, fp8._merge_att_output_kernel)
    for num_q_tokens in [1, 3, 8, 16, 17, 33, 64]:
        output = torch.randn(num_q_tokens, 4, 512, dtype=torch.float32, device="cuda")
        lse = torch.randn(4, num_q_tokens, dtype=torch.float32, device="cuda")
        chunk_output = torch.randn_like(output)
        chunk_lse = (
            torch.randn(num_q_tokens, 4, dtype=torch.float32, device="cuda").T
            if transposed_lse
            else torch.randn_like(lse)
        )
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
            q_head_num=4,
            q_token_num=num_q_tokens,
            BLOCK_HEADS=4,
            num_warps=4,
        )
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-6)
    assert len(hashes) == 7 and len(set(hashes)) == 1


def test_prefill_kv_gather_reuses_kernel_across_token_counts(monkeypatch):
    hashes = record_kernel_hashes(monkeypatch, fp8._gather_prefill_kv_kernel)
    cached_kv = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
    packed_kv = torch.zeros(64, 1, 584, dtype=torch.uint8, device="cuda")
    packed_kv[:, 0, :448] = cached_kv[:, :448].to(torch.float8_e4m3fn).view(torch.uint8)
    packed_kv[:, 0, 448:576] = cached_kv[:, 448:].contiguous().view(torch.uint8)
    packed_kv[:, 0, 576:583] = 127
    cached_kv[:, :448] = cached_kv[:, :448].to(torch.float8_e4m3fn).to(torch.bfloat16)
    table = torch.arange(64, dtype=torch.int32, device="cuda").view(1, -1)
    requests = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    for num_tokens in [1, 3, 8, 16, 17, 32, 33]:
        cu_kv = torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda")
        prefill_kv = torch.randn(1, 1, 512, dtype=torch.bfloat16, device="cuda")
        output = torch.empty(num_tokens, 1, 512, dtype=torch.bfloat16, device="cuda")
        fp8._gather_prefill_kv_kernel[(num_tokens // 4 + 1,)](
            packed_kv,
            prefill_kv,
            output,
            table,
            requests,
            cu_kv,
            cu_q,
            0,
            num_tokens,
            packed_kv.stride(0),
            prefill_kv.stride(0),
            prefill_kv.stride(2),
            req_to_token_indexs_stride_b=table.stride(0),
            req_to_token_indexs_stride_s=table.stride(1),
            batch_size=1,
            BLOCK_ROWS=4,
            num_warps=4,
        )
        expected = cached_kv[:num_tokens].clone()
        expected[-1] = prefill_kv[0, 0]
        torch.testing.assert_close(output[:, 0], expected, rtol=0, atol=0)
    assert len(hashes) == 7 and len(set(hashes)) == 1
