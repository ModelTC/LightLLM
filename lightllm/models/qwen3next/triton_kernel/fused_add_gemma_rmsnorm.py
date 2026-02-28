import torch

import triton
import triton.language as tl

from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _fused_add_gemma_rmsnorm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    x_stride0,
    x_stride1,
    r_stride0,
    r_stride1,
    y_stride0,
    y_stride1,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused in-place residual add + Gemma RMSNorm.

    For each row:
      1. sum = x + residual          (written back to x in-place)
      2. rstd = 1 / sqrt(mean(sum²) + eps)
      3. y = sum * rstd * (w + 1.0)  (Gemma-style)
    """
    row = tl.program_id(0)
    x_ptr = x_ptr + row * x_stride0
    r_ptr = r_ptr + row * r_stride0
    y_ptr = y_ptr + row * y_stride0

    # Pass 1: compute sum = x + residual, write back to x, accumulate sum² for variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols * x_stride1, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + cols * r_stride1, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        # Write sum back to x (in-place residual add)
        tl.store(x_ptr + cols * x_stride1, s.to(x_ptr.dtype.element_ty), mask=mask)
        _var += s * s

    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Pass 2: normalize and apply Gemma-style linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # Re-read x (now contains sum); hot in L2 from the write in pass 1
        s = tl.load(x_ptr + cols * x_stride1, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        y = s * rstd * (w + 1.0)
        tl.store(y_ptr + cols * y_stride1, y.to(y_ptr.dtype.element_ty), mask=mask)


def _get_fused_add_gemma_rmsnorm_configs():
    """Generate configurations for autotuning fused add + Gemma RMSNorm kernel."""
    configs = []
    for block_size in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 65536 * 2]:
        for num_warps in [1, 2, 4, 8]:
            configs.append({"BLOCK_SIZE": block_size, "num_warps": num_warps, "num_stages": 1})
    return configs


def _get_fused_add_gemma_rmsnorm_static_key(x: torch.Tensor, w: torch.Tensor):
    """Generate static key for caching autotuned configurations."""
    N = x.shape[-1]
    return {
        "x_dtype": str(x.dtype),
        "weight_dtype": str(w.dtype),
        "N": N,
    }


@autotune(
    kernel_name="fused_add_gemma_rmsnorm:v1",
    configs_gen_func=_get_fused_add_gemma_rmsnorm_configs,
    static_key_func=_get_fused_add_gemma_rmsnorm_static_key,
    run_key_func=lambda x: x.shape[-1],
    mutates_args=["x"],
)
def fused_add_gemma_rmsnorm(x, residual, w, eps, out=None, run_config: dict = None):
    """Fused in-place residual add + Gemma RMSNorm.

    x: [M, N] - modified in-place (x += residual)
    residual: [M, N] - residual to add (will be viewed as [-1, N])
    w: [N] - norm weight (Gemma-style: applies w + 1.0)
    eps: float
    out: [M, N] - output buffer (allocated if None)
    Returns: out
    """
    N = x.shape[-1]
    y = torch.empty_like(x) if out is None else out
    x_arg = x.view(-1, N)
    r_arg = residual.view(-1, N)
    y_arg = y.view(-1, N)

    M = x_arg.shape[0]

    # Default heuristic when autotune is disabled or no config provided
    if not run_config:
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This fused_add_gemma_rmsnorm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        run_config = {"BLOCK_SIZE": BLOCK_SIZE, "num_warps": num_warps, "num_stages": 1}

    BLOCK_SIZE = run_config["BLOCK_SIZE"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    _fused_add_gemma_rmsnorm_kernel[(M,)](
        x_arg,
        r_arg,
        w,
        y_arg,
        x_stride0=x_arg.stride(0),
        x_stride1=x_arg.stride(1),
        r_stride0=r_arg.stride(0),
        r_stride1=r_arg.stride(1),
        y_stride0=y_arg.stride(0),
        y_stride1=y_arg.stride(1),
        N=N,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y


def _fused_add_gemma_rmsnorm_torch(x, residual, weight, eps):
    """Reference implementation for correctness testing."""
    original_dtype = x.dtype
    x = x.to(torch.float32)
    residual = residual.to(torch.float32)
    s = x + residual
    normed = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    out = normed * (1.0 + weight.float())
    return s.to(original_dtype), out.to(original_dtype)


def test_fused_add_gemma_rmsnorm(M=128, N=2048, dtype=torch.bfloat16, eps=1e-5, device="cuda"):
    """Verify fused kernel matches separate add + gemma_rmsnorm."""
    x_shape = (M, N)
    w_shape = (N,)
    weight = torch.rand(w_shape, dtype=dtype, device=device)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    residual = 0.1 * torch.randn(x_shape, dtype=dtype, device=device)

    # Clone x for reference (since fused modifies x in-place)
    x_ref = x.clone()
    x_fused = x.clone()

    # Reference: separate add + norm
    x_ref_sum, y_ref = _fused_add_gemma_rmsnorm_torch(x_ref, residual, weight, eps)

    # Fused kernel
    y_fused = fused_add_gemma_rmsnorm(x_fused, residual, weight, eps)

    # Check x was modified in-place (x += residual)
    print(f"Test: M={M}, N={N}, dtype={dtype}")
    print(f"  x in-place max delta: {torch.max(torch.abs(x_fused - x_ref_sum)):.6e}")
    print(f"  output max delta:     {torch.max(torch.abs(y_fused - y_ref)):.6e}")

    atol = 1e-2 if dtype == torch.float32 else 5e-2
    assert torch.allclose(x_fused, x_ref_sum, atol=atol, rtol=0), "x in-place update mismatch!"
    assert torch.allclose(y_fused, y_ref, atol=atol, rtol=0), "output mismatch!"
    print("  PASSED")


if __name__ == "__main__":
    test_fused_add_gemma_rmsnorm(M=1, N=2048)
    test_fused_add_gemma_rmsnorm(M=128, N=2048)
    test_fused_add_gemma_rmsnorm(M=1, N=2048, dtype=torch.float16)
    test_fused_add_gemma_rmsnorm(M=64, N=4096, dtype=torch.float32)
    print("All tests passed!")
