import pytest
import torch

from lightllm.common.basemodel.triton_kernel.att.prefill_att.context_flashattention_nopad import (
    context_attention_fwd,
)


@pytest.mark.parametrize("causal", [True, False])
def test_context_attention_mask_mode_with_prefix_cache(causal):
    torch.manual_seed(7)
    q_lens = [3, 2]
    cache_lens = [1, 2]
    kv_lens = [q_len + cache_len for q_len, cache_len in zip(q_lens, cache_lens)]

    q = torch.randn(sum(q_lens), 2, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(16, 1, 32, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    req_to_tokens = torch.zeros(2, 8, device="cuda", dtype=torch.int32)
    req_to_tokens[0, : kv_lens[0]] = torch.tensor([3, 7, 1, 9], device="cuda", dtype=torch.int32)
    req_to_tokens[1, : kv_lens[1]] = torch.tensor([2, 8, 5, 11], device="cuda", dtype=torch.int32)
    b_req_idx = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    b_start_loc = torch.tensor([0, q_lens[0]], device="cuda", dtype=torch.int32)
    b_seq_len = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    b_cache_len = torch.tensor(cache_lens, device="cuda", dtype=torch.int32)

    output = torch.empty_like(q)
    context_attention_fwd(
        q,
        k,
        v,
        output,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_cache_len,
        max(q_lens),
        req_to_tokens,
        causal=causal,
        use_ieee_fp32_attention=True,
    )

    reference = []
    q_offset = 0
    for batch_index, (q_len, cache_len, kv_len) in enumerate(zip(q_lens, cache_lens, kv_lens)):
        pages = req_to_tokens[batch_index, :kv_len].long()
        batch_k = k[pages, 0]
        batch_v = v[pages, 0]
        for query_index in range(q_len):
            visible_kv_len = cache_len + query_index + 1 if causal else kv_len
            scores = torch.einsum("hd,sd->hs", q[q_offset + query_index], batch_k[:visible_kv_len]) / (
                q.shape[-1] ** 0.5
            )
            probabilities = scores.softmax(dim=-1)
            reference.append(torch.einsum("hs,sd->hd", probabilities, batch_v[:visible_kv_len]))
        q_offset += q_len

    torch.testing.assert_close(output, torch.stack(reference), atol=2e-4, rtol=2e-4)
