import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.fused_recurrent import (
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


@pytest.mark.parametrize("per_channel,lower_bound", [(False, None), (True, None), (True, -5.0)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_decode_fused_and_precomputed_gates_match_reference(per_channel, lower_bound, state_dtype):
    """Exercise scalar/vector gates, strided projections, and separate state slots."""
    torch.manual_seed(53)
    batch, H, HV, K, V = 4, 2, 4, 128, 64
    gate_dim = HV * K if per_channel else HV
    widths = [H * K, H * K, HV * V, gate_dim, HV]
    mixed = torch.randn(batch, sum(widths), device="cuda", dtype=torch.bfloat16)
    q, k, v, a, b = mixed.split(widths, dim=-1)
    q, k, v = q.view(batch, 1, H, K), k.view(batch, 1, H, K), v.view(batch, 1, HV, V)
    if per_channel:
        a = a.view(batch, HV, K)
    A_log = torch.randn(HV, device="cuda") * 0.1
    bias = torch.randn(a.shape[1:], device="cuda") * 0.1
    amplitude = A_log.exp().view(HV, 1) if per_channel else A_log.exp()
    x = a.float() + bias
    if lower_bound is None:
        g = -amplitude * torch.nn.functional.softplus(x)
    else:
        g = lower_bound * torch.sigmoid(amplitude * x)
    beta = b.float().sigmoid()
    decay = g.exp().unsqueeze(-1) if per_channel else g.exp()[..., None, None]
    q_ref = q[:, 0].float()
    k_ref = k[:, 0].float()
    q_ref = (q_ref * torch.rsqrt(q_ref.square().sum(-1, keepdim=True) + 1e-6) / K ** 0.5).repeat_interleave(
        HV // H, dim=1
    )
    k_ref = (k_ref * torch.rsqrt(k_ref.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(HV // H, dim=1)
    state_ref = torch.randn(12, HV, K, V, device="cuda", dtype=state_dtype) * 0.1
    state_fused, state_precomputed = state_ref.clone(), state_ref.clone()
    read_idx = torch.tensor([4, 1, 7, 3], device="cuda", dtype=torch.int32)
    write_idx = torch.tensor([2, 9, 0, 6], device="cuda", dtype=torch.int32)
    for _ in range(3):
        state = state_ref[read_idx].float() * decay
        delta = (v[:, 0].float() - torch.einsum("bhkv,bhk->bhv", state, k_ref)) * beta[..., None]
        state += k_ref[..., None] * delta[..., None, :]
        expected = torch.einsum("bhkv,bhk->bhv", state, q_ref).unsqueeze(1)
        state_ref[write_idx] = state.to(state_dtype)
        for cache, gates in (
            (state_fused, dict(A_log=A_log, dt_bias=bias, a_raw=a, b_raw=b, lower_bound=lower_bound)),
            (state_precomputed, dict(g=g.unsqueeze(1), beta=beta.unsqueeze(1))),
        ):
            output, _ = fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                initial_state=cache,
                ssm_state_indices=read_idx,
                ssm_state_write_indices=write_idx,
                use_qk_l2norm_in_kernel=True,
                **gates,
            )
            torch.testing.assert_close(output.float(), expected, atol=5e-4, rtol=1e-2)
            # Comparing the entire cache also checks that unrelated slots stay intact.
            torch.testing.assert_close(cache, state_ref, atol=5e-4, rtol=1e-2)
        read_idx, write_idx = write_idx, read_idx


# NOTE: the decode-only `cu_seqlens is None` contract from upstream #1349 was
# intentionally lifted on this branch so the Qwen3Next MTP verify path can drive
# the kernel with variable-length verify chunks (cu_seqlens + 2D SSM index
# rows). That varlen path is exercised end-to-end by the MTP GSM8K accuracy
# check rather than a hand-rolled unit test.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
