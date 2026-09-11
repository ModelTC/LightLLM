import pytest
import torch

from lightllm.common.basemodel.attention.split_prefill import (
    fixed_suffix_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("has_state", "suffix_length"), [(True, 7), (False, 6)])
def test_fixed_suffix_attention_matches_dense_reference(has_state, suffix_length):
    torch.manual_seed(7)
    batch_size, query_heads, head_dim = 2, 4, 32
    token_count = batch_size * suffix_length
    query = torch.randn(
        token_count,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn(token_count, 1, head_dim, dtype=query.dtype, device="cuda")
    value = torch.randn_like(key)

    output, lse = fixed_suffix_attention(
        query,
        key,
        value,
        batch_size=batch_size,
        suffix_length=suffix_length,
        first_suffix_token_isolated=has_state,
    )

    query_ref = query.view(batch_size, suffix_length, query_heads, head_dim).float()
    key_ref = key.view(batch_size, suffix_length, 1, head_dim).float()
    value_ref = value.view(batch_size, suffix_length, 1, head_dim).float()
    scores = torch.einsum("bqhd,bkhd->bqhk", query_ref, key_ref)
    scores.mul_(head_dim**-0.5)
    if has_state:
        scores[:, 0, :, 1:] = -torch.inf
    expected_lse = scores.logsumexp(-1).reshape(token_count, query_heads)
    expected = torch.einsum(
        "bqhk,bkhd->bqhd",
        scores.softmax(-1),
        value_ref,
    ).reshape_as(output)

    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.02)
