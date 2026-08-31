import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda import kda_safe_gate


if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("sequence_length", [3, 6])
def test_fused_kda_gate_matches_materialized_gate(sequence_length):
    torch.manual_seed(2026 + sequence_length)
    request_count, query_heads, value_heads, key_dim, value_dim = 3, 2, 4, 64, 64
    token_count = request_count * sequence_length
    slot_count = token_count + 8

    q = torch.randn(1, token_count, query_heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, token_count, value_heads, value_dim, device="cuda", dtype=torch.bfloat16)
    raw_gate = torch.randn(1, token_count, value_heads * key_dim, device="cuda", dtype=torch.bfloat16)
    raw_beta = torch.randn(1, token_count, value_heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(value_heads, device="cuda", dtype=torch.float32) * 0.1
    gate_bias = torch.randn(value_heads * key_dim, device="cuda", dtype=torch.float32) * 0.1
    state = torch.randn(
        slot_count,
        value_heads,
        key_dim,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_indices = torch.arange(token_count, device="cuda", dtype=torch.int32).view(request_count, sequence_length)
    cu_seqlens = torch.arange(request_count + 1, device="cuda", dtype=torch.int64) * sequence_length
    accepted = torch.full((request_count,), sequence_length, device="cuda", dtype=torch.int32)

    reference_state = state.clone()
    reference, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=kda_safe_gate(raw_gate, a_log, gate_bias),
        beta=raw_beta.float().sigmoid(),
        initial_state=reference_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        ssm_state_write_indices=state_indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
    )

    fused_state = state.clone()
    fused, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        initial_state=fused_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        ssm_state_write_indices=state_indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        A_log=a_log,
        dt_bias=gate_bias,
        a_raw=raw_gate.reshape(token_count, value_heads * key_dim),
        b_raw=raw_beta.reshape(token_count, value_heads),
        is_kda=True,
        kda_lower_bound=-5.0,
    )

    torch.testing.assert_close(fused, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(fused_state, reference_state, rtol=2e-3, atol=2e-3)
