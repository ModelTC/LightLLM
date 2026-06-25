"""Correctness: fused silu_and_mul + group fp8 quant vs unfused silu_and_mul_fwd then per_token_group_quant_fp8."""
import torch
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul_group_quant import silu_and_mul_group_quant_fwd


def dequant(q, s, group_size):
    # q: [M, N] fp8, s: [M, N//group] fp32 -> [M, N] fp32
    M, N = q.shape
    qf = q.to(torch.float32).reshape(M, N // group_size, group_size)
    return (qf * s.reshape(M, N // group_size, 1)).reshape(M, N)


def check(M, N, group_size=128, layout="blocked", dtype=torch.bfloat16):
    torch.manual_seed(0)
    inp = torch.randn(M, 2 * N, dtype=dtype, device="cuda").contiguous()

    # unfused reference
    tmp = torch.empty(M, N, dtype=dtype, device="cuda")
    silu_and_mul_fwd(inp, tmp, layout=layout)
    q_ref, s_ref = per_token_group_quant_fp8(tmp, group_size, dtype=torch.float8_e4m3fn)

    # fused
    q = torch.empty(M, N, dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(M, N // group_size, dtype=torch.float32, device="cuda")
    silu_and_mul_group_quant_fwd(inp, q, s, group_size, layout=layout)

    dq_ref = dequant(q_ref, s_ref.reshape(M, N // group_size), group_size)
    dq = dequant(q, s, group_size)
    cos = torch.nn.functional.cosine_similarity(dq.flatten(), dq_ref.flatten(), dim=0).item()
    rel = ((dq - dq_ref).abs() / (dq_ref.abs() + 1e-4)).max().item()
    s_match = (s.reshape(-1) - s_ref.reshape(-1)).abs().max().item()
    q_exact = (q.to(torch.float32) == q_ref.to(torch.float32)).float().mean().item()
    print(
        f"M={M:<5} N={N:<6} {layout:<11} cos={cos:.7f} dequant_max_rel={rel:.3e} "
        f"scale_max_abs={s_match:.3e} q_exact_frac={q_exact:.4f}"
    )
    assert cos > 0.9999, f"cosine too low: {cos}"
    assert s_match < 1e-6 or q_exact > 0.99, "scale/quant diverged beyond fp8 rounding"


if __name__ == "__main__":
    print("=== correctness (fused silu+group-quant vs unfused) ===")
    for M in [1, 8, 16, 64, 256]:
        check(M, 1536)  # glm/deepseek-ish moe intermediate
    check(8, 2048)
    check(13, 768)
    check(8, 1536, layout="interleaved")
    print("OK")
