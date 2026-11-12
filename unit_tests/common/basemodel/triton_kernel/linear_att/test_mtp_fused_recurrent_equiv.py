import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
    mtp_fused_recurrent_gated_delta_rule,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _run_both(
    q,
    k,
    v,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_write_indices,
    num_accepted_tokens,
    A_log,
    dt_bias,
    a_raw,
    b_raw,
):
    """Run old (via autograd.Function) and new (direct kernel) side-by-side."""
    state_old = initial_state.clone()
    state_new = initial_state.clone()

    o_old, fs_old = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        initial_state=state_old,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_write_indices=ssm_state_write_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
    )

    o_new, fs_new = mtp_fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        initial_state=state_new,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_write_indices=ssm_state_write_indices,
        num_accepted_tokens=num_accepted_tokens,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
    )

    return o_old, o_new, fs_old, fs_new


@pytest.mark.parametrize("mtp_step", [1, 2, 3])
def test_mtp_verify_path(mtp_step):
    """MTP verify path with cu_seqlens and 2D SSM indices."""
    torch.manual_seed(123)
    batch, H, HV, K, V = 2, 2, 8, 64, 64
    seqlen = mtp_step + 1
    num_tokens = batch * seqlen
    cache_slots = 64

    q = torch.randn(1, num_tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, num_tokens, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, num_tokens, HV, V, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    a_raw = torch.randn(num_tokens, HV, device="cuda", dtype=torch.bfloat16)
    b_raw = torch.randn(num_tokens, HV, device="cuda", dtype=torch.bfloat16)
    ssm_state = torch.randn(cache_slots, HV, K, V, device="cuda", dtype=torch.bfloat16)
    ssm_idx = torch.randint(0, cache_slots, (batch, seqlen), device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * seqlen
    num_accepted = torch.full((batch,), seqlen, device="cuda", dtype=torch.int32)

    o_old, o_new, fs_old, fs_new = _run_both(
        q,
        k,
        v,
        ssm_state,
        cu_seqlens.to(torch.long),
        ssm_idx,
        ssm_idx,
        num_accepted,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
    )

    assert torch.equal(
        o_old, o_new
    ), f"output mismatch, max diff={torch.abs(o_old.float() - o_new.float()).max().item():.6f}"
    if not torch.equal(fs_old, fs_new):
        assert torch.allclose(fs_old.float(), fs_new.float(), rtol=1e-2, atol=5.0), (
            f"final_state mismatch at mtp_step={mtp_step}, "
            f"max diff={torch.abs(fs_old.float() - fs_new.float()).max().item():.6f}"
        )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_variable_length_cu_seqlens(seed):
    """Variable-length sequences via cu_seqlens: individual lengths vary,
    each ≤ mtp_step+1, with some padded to shorter lengths."""
    torch.manual_seed(seed)
    batch, mtp_step, H, HV, K, V = 4, 3, 2, 8, 64, 64
    max_len = mtp_step + 1  # 4
    cache_slots = 64

    # Random per-sequence lengths in [1, max_len]
    lengths = torch.randint(1, max_len + 1, (batch,), device="cpu")
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32)
    torch.cumsum(lengths, dim=0, out=cu_seqlens[1:])

    num_tokens = int(cu_seqlens[-1].item())
    q = torch.randn(1, num_tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, num_tokens, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, num_tokens, HV, V, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    a_raw = torch.randn(num_tokens, HV, device="cuda", dtype=torch.bfloat16)
    b_raw = torch.randn(num_tokens, HV, device="cuda", dtype=torch.bfloat16)
    ssm_state = torch.randn(cache_slots, HV, K, V, device="cuda", dtype=torch.bfloat16)
    # 2D indices: S+1 = max_len columns, unused columns for short seqs are irrelevant
    ssm_idx = torch.randint(0, cache_slots, (batch, max_len), device="cuda", dtype=torch.int32)
    num_accepted = lengths.to(device="cuda", dtype=torch.int32)

    o_old, o_new, fs_old, fs_new = _run_both(
        q,
        k,
        v,
        ssm_state,
        cu_seqlens.to(torch.long).cuda(),
        ssm_idx,
        ssm_idx,
        num_accepted,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
    )

    assert torch.equal(
        o_old, o_new
    ), f"seed={seed}: output mismatch, max diff={torch.abs(o_old.float() - o_new.float()).max().item():.6f}"
    if not torch.equal(fs_old, fs_new):
        assert torch.allclose(fs_old.float(), fs_new.float(), rtol=1e-2, atol=5.0), (
            f"seed={seed}: final_state mismatch, "
            f"max diff={torch.abs(fs_old.float() - fs_new.float()).max().item():.6f}"
        )


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_match_decode_preserves_every_sequential_snapshot(state_dtype, steps):
    torch.manual_seed(3107)
    batch, heads, value_heads, width = 2, 4, 12, 128
    count = batch * steps
    q = torch.randn(batch, steps, heads, width, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, steps, value_heads, width, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(batch, steps, value_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    indices = torch.arange(count, device="cuda", dtype=torch.int32).view(batch, steps)
    accepted = torch.tensor([1, steps], device="cuda", dtype=torch.int32)
    initial = torch.randn(count, value_heads, width, width, device="cuda", dtype=state_dtype) * 0.1
    state = initial[indices.gather(1, accepted.long()[:, None] - 1).flatten().long()].clone()
    expected_states = initial.clone()
    gates = dict(A_log=torch.zeros(value_heads, device="cuda"), dt_bias=torch.zeros(value_heads, device="cuda"))
    outputs = []
    for step in range(steps):
        output, _ = fused_recurrent_gated_delta_rule(
            q=q[:, step : step + 1],
            k=k[:, step : step + 1],
            v=v[:, step : step + 1],
            initial_state=state,
            inplace_final_state=True,
            ssm_state_indices=torch.arange(batch, device="cuda", dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
            a_raw=a[:, step],
            b_raw=b[:, step],
            **gates,
        )
        outputs.append(output)
        expected_states.index_copy_(0, indices[:, step].long(), state)
    actual, actual_states = mtp_fused_recurrent_gated_delta_rule(
        q=q.reshape(1, count, heads, width),
        k=k.reshape(1, count, heads, width),
        v=v.reshape(1, count, value_heads, width),
        initial_state=initial.clone(),
        cu_seqlens=torch.arange(batch + 1, device="cuda", dtype=torch.int64) * steps,
        ssm_state_indices=indices,
        ssm_state_write_indices=indices,
        num_accepted_tokens=accepted,
        a_raw=a.reshape(count, value_heads),
        b_raw=b.reshape(count, value_heads),
        run_config={"BV": 32, "num_warps": 4, "num_stages": 1},
        **gates,
    )
    expected = torch.cat(outputs, dim=1).reshape_as(actual)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_states, expected_states, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
