import pytest
import torch

from lightllm.models.qwen3next.triton_kernel.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from lightllm.models.qwen3next.triton_kernel.mtp_fused_recurrent import (
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
