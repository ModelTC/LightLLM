import pytest
import torch

from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("batch", [1, 2, 16])
def test_decode_strided_views_match_contiguous(batch):
    """q/k/v/a/b passed as column views of one projection output (the decode
    path layout) must produce the same result as contiguous copies."""
    torch.manual_seed(0)
    H, HV, K, V = 2, 8, 128, 128
    key_dim, value_dim = H * K, HV * V
    qkv_dim = 2 * key_dim + value_dim
    total_dim = qkv_dim + value_dim + 2 * HV  # qkv + z + b + a
    cache_slots = 64

    mixed = torch.randn(batch, total_dim, device="cuda", dtype=torch.bfloat16)
    mixed_qkv = mixed[:, :qkv_dim]
    b_raw = mixed[:, qkv_dim + value_dim : qkv_dim + value_dim + HV]
    a_raw = mixed[:, qkv_dim + value_dim + HV :]

    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = query.view(batch, 1, H, K)
    k = key.view(batch, 1, H, K)
    v = value.view(batch, 1, HV, V)

    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    ssm_state = torch.randn(cache_slots, HV, K, V, device="cuda", dtype=torch.bfloat16)
    idx = torch.randperm(cache_slots, device="cuda")[:batch].to(torch.int32)

    def run(q_, k_, v_, a_, b_, state):
        out, _ = fused_recurrent_gated_delta_rule(
            q=q_,
            k=k_,
            v=v_,
            initial_state=state,
            inplace_final_state=True,
            ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            a_raw=a_,
            b_raw=b_,
        )
        return out

    state_ref = ssm_state.clone()
    out_ref = run(q.contiguous(), k.contiguous(), v.contiguous(), a_raw.contiguous(), b_raw.contiguous(), state_ref)
    state_strided = ssm_state.clone()
    out_strided = run(q, k, v, a_raw, b_raw, state_strided)

    assert torch.equal(out_ref, out_strided)
    assert torch.equal(state_ref, state_strided)


# NOTE: the decode-only `cu_seqlens is None` contract from upstream #1349 was
# intentionally lifted on this branch so the Qwen3Next MTP verify path can drive
# the kernel with variable-length verify chunks (cu_seqlens + 2D SSM index
# rows). That varlen path is exercised end-to-end by the MTP GSM8K accuracy
# check rather than a hand-rolled unit test.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
