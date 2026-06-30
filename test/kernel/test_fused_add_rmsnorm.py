"""Correctness + microbench for fused_add_rmsnorm vs the production (add_ ; rmsnorm) sequence."""
import torch
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.fused_add_rmsnorm import fused_add_rmsnorm_forward


def ref(residual, x, weight, eps):
    # exactly what token_forward does today: in-place residual add, then a separate rmsnorm
    res = residual.clone()
    res.add_(x)
    out = rmsnorm_forward(res, weight, eps)
    return out, res


def check(M, N, dtype, eps=1e-5):
    torch.manual_seed(0)
    residual = (-2.3 + 0.5 * torch.randn(M, N, dtype=dtype, device="cuda")).contiguous()
    x = (0.7 * torch.randn(M, N, dtype=dtype, device="cuda")).contiguous()
    weight = torch.rand(N, dtype=dtype, device="cuda")

    res_ref0 = residual.clone()
    out_ref, res_ref = ref(res_ref0, x, weight, eps)

    res_fused = residual.clone()
    out_fused = torch.empty_like(res_fused)
    fused_add_rmsnorm_forward(res_fused, x, weight, eps, out=out_fused)

    # residual update must be bit-identical to a plain bf16 add_
    res_match = torch.equal(res_fused, res_ref)
    out_abs = (out_fused.float() - out_ref.float()).abs()
    rel = out_abs / (out_ref.float().abs() + 1e-6)
    cos = torch.nn.functional.cosine_similarity(out_fused.float().flatten(), out_ref.float().flatten(), dim=0).item()
    print(
        f"M={M:<4} N={N:<6} {str(dtype):<14} residual_bitmatch={res_match} "
        f"out_max_abs={out_abs.max().item():.3e} out_max_rel={rel.max().item():.3e} cos={cos:.7f}"
    )
    assert res_match, "residual (residual+x) must bit-match a plain add_"
    # variance is now taken from the bf16-rounded sum, matching the unfused path bit-for-bit
    assert torch.equal(out_fused, out_ref), "normalized output must bit-match the unfused path"


def bench(M, N, dtype=torch.bfloat16, eps=1e-5, iters=200):
    residual = torch.randn(M, N, dtype=dtype, device="cuda")
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.rand(N, dtype=dtype, device="cuda")
    out = torch.empty_like(residual)

    def run_unfused():
        residual.add_(x)
        rmsnorm_forward(residual, weight, eps, out=out)

    def run_fused():
        fused_add_rmsnorm_forward(residual, x, weight, eps, out=out)

    for fn, name in [(run_unfused, "add_+rmsnorm"), (run_fused, "fused_add_rmsnorm")]:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        print(f"  {name:<20} {start.elapsed_time(end) / iters * 1e3:.2f} us/call")


if __name__ == "__main__":
    print("=== correctness ===")
    for M in [1, 2, 4, 8, 16, 32]:
        check(M, 6144, torch.bfloat16)  # GLM-5.2 hidden
    check(1, 7168, torch.bfloat16)  # DeepSeek-V3 hidden
    check(1, 4096, torch.float16)
    check(13, 6144, torch.bfloat16)
    print("=== microbench (decode-shaped, bs in 1..32 @ N=6144) ===")
    for M in [1, 4, 16, 32]:
        print(f"M={M}:")
        bench(M, 6144)
    print("OK")
