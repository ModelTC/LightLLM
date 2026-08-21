"""Correctness (+ __main__ microbench) for fused_add_rmsnorm vs the unfused (add_ ; rmsnorm) sequence."""
import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.fused_add_rmsnorm import fused_add_rmsnorm_forward


def _ref(residual, x, weight, eps):
    # exactly what token_forward does today: in-place residual add, then a separate rmsnorm
    res = residual.clone()
    res.add_(x)
    out = rmsnorm_forward(res, weight, eps)
    return out, res


@pytest.mark.parametrize(
    "M, N, dtype",
    [(M, 6144, torch.bfloat16) for M in [1, 2, 4, 8, 16, 32]]  # GLM-5.2 hidden
    + [
        (1, 7168, torch.bfloat16),  # DeepSeek-V3 hidden
        (1, 4096, torch.float16),
        (13, 6144, torch.bfloat16),
    ],
)
def test_fused_add_rmsnorm_matches_unfused(M, N, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    eps = 1e-5
    torch.manual_seed(0)
    residual = (-2.3 + 0.5 * torch.randn(M, N, dtype=dtype, device="cuda")).contiguous()
    x = (0.7 * torch.randn(M, N, dtype=dtype, device="cuda")).contiguous()
    weight = torch.rand(N, dtype=dtype, device="cuda")

    out_ref, res_ref = _ref(residual, x, weight, eps)

    res_fused = residual.clone()
    out_fused = torch.empty_like(res_fused)
    fused_add_rmsnorm_forward(res_fused, x, weight, eps, out=out_fused)

    # residual update must be bit-identical to a plain add_
    assert torch.equal(res_fused, res_ref), "residual (residual+x) must bit-match a plain add_"
    # variance is taken from the rounded sum, matching the unfused path bit-for-bit
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
    print("=== microbench (decode-shaped, bs in 1..32 @ N=6144) ===")
    for M in [1, 4, 16, 32]:
        print(f"M={M}:")
        bench(M, 6144)
    pytest.main([__file__, "-q"])
