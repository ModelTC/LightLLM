import torch

import triton
import triton.language as tl
import os

from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _gemma_rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    x_stride0,
    x_stride1,
    y_stride0,
    y_stride1,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * x_stride0
    y_ptr = y_ptr + row * y_stride0

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols * x_stride1, mask=cols < N, other=0.0).to(tl.float32)
        _sum += x * x

    var = tl.sum(_sum, axis=0) / N
    rstd = 1 / tl.sqrt(var + EPS)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + cols * x_stride1, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        w = w + 1.0
        y = x_hat * w
        # Write output
        tl.store(y_ptr + cols * y_stride1, y.to(y_ptr.dtype.element_ty), mask=mask)


def _get_gemma_rmsnorm_configs():
    """Generate configurations for autotuning gemma RMSNorm kernel."""
    configs = []
    # Different BLOCK_SIZE values (powers of 2)
    for block_size in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 65536 * 2]:
        # Different number of warps
        for num_warps in [1, 2, 4, 8]:
            for num_stages in [1, 2, 3, 4, 5]:
                configs.append({"BLOCK_SIZE": block_size, "num_warps": num_warps, "num_stages": num_stages})
    return configs


def _get_gemma_rmsnorm_static_key(x: torch.Tensor, w: torch.Tensor):
    """Generate static key for caching autotuned configurations."""
    N = x.shape[-1]
    return {
        "x_dtype": str(x.dtype),
        "weight_dtype": str(w.dtype),
        "N": N,
    }


@autotune(
    kernel_name="gemma_rmsnorm_forward:v1",
    configs_gen_func=_get_gemma_rmsnorm_configs,
    static_key_func=_get_gemma_rmsnorm_static_key,
    run_key_func=lambda x: x.shape[-1],
)
def gemma_rmsnorm_forward(x, w, eps, out=None, run_config: dict = None):
    # Inplace gemma RMS Norm
    # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    N = x.shape[-1]
    y = torch.empty_like(x) if out is None else out
    x_arg = x.view(-1, N)
    y_arg = y.view(-1, N)

    M, _ = x_arg.shape

    # Default heuristic when autotune is disabled or no config provided
    if not run_config:
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This gemma rmsnorm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        run_config = {"BLOCK_SIZE": BLOCK_SIZE, "num_warps": num_warps, "num_stages": 1}

    BLOCK_SIZE = run_config["BLOCK_SIZE"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    _gemma_rmsnorm_fwd_kernel[(M,)](
        x_arg,
        w,
        y_arg,
        x_stride0=x.stride(0),
        x_stride1=x.stride(1),
        y_stride0=y.stride(0),
        y_stride1=y.stride(1),
        N=N,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y


def _gemma_rmsnorm_fwd_torch(x, weight, eps):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * (1.0 + weight.float())
    return x.to(original_dtype)


def test_rms_norm(M, N, dtype, eps=1e-5, device="cuda"):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    # forward pass
    y_tri = gemma_rmsnorm_forward(x, weight, eps)
    y_ref = _gemma_rmsnorm_fwd_torch(x, weight, eps)

    # compare
    print("type:", y_tri.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    # Use appropriate tolerance based on dtype
    atol = 1e-2 if dtype == torch.float32 else 5e-2
    assert torch.allclose(y_tri, y_ref, atol=atol, rtol=0)
    return
