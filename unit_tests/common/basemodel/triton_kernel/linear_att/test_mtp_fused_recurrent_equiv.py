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
    """Compare fused verification with sequential single-token decode.

    Decode reloads the persistent state dtype after every token. Running the
    old kernel over an entire sequence instead keeps intermediate states in
    FP32, which is a different recurrence for BF16 state storage.
    """
    state_old = initial_state.clone()
    state_new = initial_state.clone()
    o_old = torch.empty_like(v)
    boundaries = cu_seqlens.tolist()
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        accepted = int(num_accepted_tokens[row])
        read_index = ssm_state_indices[row : row + 1, accepted - 1 : accepted]
        for step, token in enumerate(range(start, end)):
            write_index = ssm_state_write_indices[row : row + 1, step : step + 1]
            output, _ = fused_recurrent_gated_delta_rule(
                q=q[:, token : token + 1],
                k=k[:, token : token + 1],
                v=v[:, token : token + 1],
                initial_state=state_old,
                inplace_final_state=True,
                ssm_state_indices=read_index,
                ssm_state_write_indices=write_index,
                use_qk_l2norm_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                a_raw=a_raw[token : token + 1],
                b_raw=b_raw[token : token + 1],
            )
            o_old[:, token : token + 1].copy_(output)
            read_index = write_index

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
        # Match the single-token decode reduction layout for bitwise comparison.
        # Other layouts, including the default, have an independent torch
        # reference in test_mtp_fused_recurrent_autotune.py.
        run_config={"BV": 32, "num_warps": 4, "num_stages": 1},
    )

    return o_old, o_new, state_old, fs_new


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
    ssm_idx = torch.randperm(cache_slots, device="cuda", dtype=torch.int32)[: batch * seqlen].view(batch, seqlen)
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
    torch.testing.assert_close(fs_old, fs_new, rtol=0, atol=0)


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
    ssm_idx = torch.randperm(cache_slots, device="cuda", dtype=torch.int32)[: batch * max_len].view(batch, max_len)
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
    torch.testing.assert_close(fs_old, fs_new, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
