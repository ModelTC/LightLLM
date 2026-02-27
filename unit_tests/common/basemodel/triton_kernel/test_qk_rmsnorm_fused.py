import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.qk_norm import (
    qk_rmsnorm_fused_forward,
    qk_rmsnorm_forward,
)


def torch_qk_rmsnorm(q, w_q, eps=1e-6):
    input_dtype = q.dtype
    head_dim = w_q.shape[0]
    q_fp32 = q.to(torch.float32)
    q_fp32 = q_fp32.view(-1, head_dim)
    variance = q_fp32.pow(2).mean(dim=-1, keepdim=True)
    q_fp32 = q_fp32 * torch.rsqrt(variance + eps)
    return (q_fp32 * w_q.to(input_dtype)).view_as(q)


def test_qk_rmsnorm_fused_matches_reference():
    """Compare fused QK RMSNorm with separate reference RMSNorm kernels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for qk_rmsnorm_fused test")

    torch.manual_seed(0)

    # 模拟配置: Batch=2, Seq=128, Head_Dim=128
    # Q: 16 Heads, K: 4 Heads (GQA 场景)
    B, S, D = 2, 128, 128
    H_Q = 16
    H_K = 4

    q = torch.randn((B * S, H_Q * D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B * S, H_K * D), device="cuda", dtype=torch.bfloat16)

    w_q = torch.ones((D,), device="cuda", dtype=torch.bfloat16)
    w_k = torch.ones((D,), device="cuda", dtype=torch.bfloat16)

    # 复制一份做对比（reference 会在新 tensor 上计算）
    q_ref = q.clone()
    k_ref = k.clone()

    # fused kernel in-place 计算
    q_out, k_out = qk_rmsnorm_fused_forward(q, k, w_q, w_k, eps=1e-6)

    # reference: 分别对 Q / K 做 RMSNorm
    q_ref_out = qk_rmsnorm_forward(q_ref, w_q, eps=1e-6)
    k_ref_out = qk_rmsnorm_forward(k_ref, w_k, eps=1e-6)
    # q_ref_out = torch_qk_rmsnorm(q_ref, w_q, eps=1e-6)
    # k_ref_out = torch_qk_rmsnorm(k_ref, w_k, eps=1e-6)

    # fused 是 in-place 的，返回的 q_out/k_out 应该与 q/k 引用一致
    assert q_out.data_ptr() == q.data_ptr()
    assert k_out.data_ptr() == k.data_ptr()

    # 误差容忍度: 由于 bfloat16 计算，设定一个合理的 atol
    q_max_diff = (q_out - q_ref_out).abs().max().item()
    k_max_diff = (k_out - k_ref_out).abs().max().item()

    print(f"Q max diff: {q_max_diff}")
    print(f"K max diff: {k_max_diff}")

    assert q_max_diff < 1e-5, f"Q max diff too large: {q_max_diff}"
    assert k_max_diff < 1e-5, f"K max diff too large: {k_max_diff}"
