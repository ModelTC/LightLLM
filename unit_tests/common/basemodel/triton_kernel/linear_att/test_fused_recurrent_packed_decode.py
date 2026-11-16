import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_decode_matches_materialized_decode():
    torch.manual_seed(17)
    batch, key_heads, value_heads = 2, 2, 4
    key_dim = value_dim = 128
    device = "cuda"
    dtype = torch.bfloat16

    mixed_qkv = torch.randn(
        batch,
        2 * key_heads * key_dim + value_heads * value_dim,
        device=device,
        dtype=dtype,
    )
    a_raw = torch.randn(batch, value_heads, device=device, dtype=dtype)
    b_raw = torch.randn_like(a_raw)
    a_log = torch.randn(value_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, device=device, dtype=torch.float32)
    state = torch.randn(batch + 1, value_heads, key_dim, value_dim, device=device, dtype=dtype)
    state_reference = state.clone()
    state_packed = state.clone()
    state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)

    q = mixed_qkv[:, : key_heads * key_dim].view(batch, 1, key_heads, key_dim)
    k = mixed_qkv[:, key_heads * key_dim : 2 * key_heads * key_dim].view(batch, 1, key_heads, key_dim)
    v = mixed_qkv[:, 2 * key_heads * key_dim :].view(batch, 1, value_heads, value_dim)

    reference, _ = fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        initial_state=state_reference,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
    )
    packed, _ = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv,
        a_raw,
        b_raw,
        a_log,
        dt_bias,
        key_dim ** -0.5,
        state_packed,
        state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(packed, reference, atol=0.02, rtol=0.02)
    torch.testing.assert_close(state_packed, state_reference, atol=0.02, rtol=0.02)
