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
    use_qk_l2norm_in_kernel,
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
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
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
    )

    return o_old, o_new, fs_old, fs_new


def _decode_meta(batch, cache_slots, device="cuda"):
    """Return (cu_seqlens, ssm_idx, ssm_widx, num_accepted) for decode."""
    cu = torch.arange(batch + 1, device=device, dtype=torch.int32)
    idx = torch.randperm(cache_slots, device=device)[:batch].to(torch.int32)
    idx_2d = idx.unsqueeze(1)
    nac = torch.ones(batch, device=device, dtype=torch.int32)
    return cu, idx_2d, idx_2d, nac


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_decode_path_fused_gating(batch):
    """Decode path with fused gating."""
    torch.manual_seed(42)
    H, HV, K, V = 2, 8, 128, 128
    cache_slots = 64

    q = torch.randn(batch, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 1, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    a_raw = torch.randn(batch, HV, device="cuda", dtype=torch.bfloat16)
    b_raw = torch.randn(batch, HV, device="cuda", dtype=torch.bfloat16)
    ssm_state = torch.randn(cache_slots, HV, K, V, device="cuda", dtype=torch.bfloat16)
    cu, idx, widx, nac = _decode_meta(batch, cache_slots)

    o_old, o_new, fs_old, fs_new = _run_both(
        q,
        k,
        v,
        ssm_state,
        cu.to(torch.long),
        idx,
        widx,
        nac,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
        True,
    )

    assert torch.equal(
        o_old, o_new
    ), f"output mismatch, max diff={torch.abs(o_old.float() - o_new.float()).max().item():.6f}"
    assert torch.equal(fs_old, fs_new), "final_state mismatch"


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
        True,
    )

    assert torch.equal(
        o_old, o_new
    ), f"output mismatch, max diff={torch.abs(o_old.float() - o_new.float()).max().item():.6f}"
    if not torch.equal(fs_old, fs_new):
        assert torch.allclose(fs_old.float(), fs_new.float(), rtol=1e-2, atol=5.0), (
            f"final_state mismatch at mtp_step={mtp_step}, "
            f"max diff={torch.abs(fs_old.float() - fs_new.float()).max().item():.6f}"
        )


@pytest.mark.parametrize("batch", [1, 2])
def test_strided_views_identical(batch):
    """Non-contiguous (strided) q/k/v produce identical results in both impls."""
    torch.manual_seed(99)
    H, HV, K, V = 2, 8, 128, 128
    key_dim, value_dim = H * K, HV * V
    qkv_dim = 2 * key_dim + value_dim
    total_dim = qkv_dim + value_dim + 2 * HV
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
    cu, idx, widx, nac = _decode_meta(batch, cache_slots)

    o_old, o_new, fs_old, fs_new = _run_both(
        q,
        k,
        v,
        ssm_state,
        cu.to(torch.long),
        idx,
        widx,
        nac,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
        True,
    )

    assert torch.equal(
        o_old, o_new
    ), f"strided output mismatch, max diff={torch.abs(o_old.float() - o_new.float()).max().item():.6f}"
    assert torch.equal(fs_old, fs_new), "strided final_state mismatch"


@pytest.mark.parametrize("without_qk_norm", [True, False])
def test_qk_l2norm_flag(without_qk_norm):
    """use_qk_l2norm_in_kernel flag behaves the same."""
    torch.manual_seed(314)
    H, HV, K, V = 2, 8, 128, 128
    batch, cache_slots = 2, 32

    q = torch.randn(batch, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 1, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    a_raw = torch.randn(batch, HV, device="cuda", dtype=torch.bfloat16)
    b_raw = torch.randn(batch, HV, device="cuda", dtype=torch.bfloat16)
    ssm_state = torch.randn(cache_slots, HV, K, V, device="cuda", dtype=torch.bfloat16)
    cu, idx, widx, nac = _decode_meta(batch, cache_slots)

    o_old, o_new, fs_old, fs_new = _run_both(
        q,
        k,
        v,
        ssm_state,
        cu.to(torch.long),
        idx,
        widx,
        nac,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
        not without_qk_norm,
    )

    assert torch.equal(o_old, o_new), (
        f"l2norm={not without_qk_norm}: output mismatch, "
        f"max diff={torch.abs(o_old.float() - o_new.float()).max().item():.6f}"
    )
    assert torch.equal(fs_old, fs_new), f"l2norm={not without_qk_norm}: final_state mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
