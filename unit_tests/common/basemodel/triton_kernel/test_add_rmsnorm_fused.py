import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import (
    rmsnorm_forward,
    add_rmsnorm_fused_forward,
)


def torch_rms_norm(x, weight, eps):
    """Reference implementation of RMSNorm using PyTorch."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class TestRMSNorm:
    """Test cases for RMSNorm kernel."""

    @pytest.mark.parametrize(
        "M, N, dtype",
        [
            (1024, 4096, torch.float16),
            (1024, 4096, torch.bfloat16),
            (512, 8192, torch.bfloat16),
            (1024, 16384, torch.bfloat16),
            (1024, 32760, torch.bfloat16),
        ],
    )
    def test_rmsnorm_correctness(self, M, N, dtype):
        """Test RMSNorm output correctness against PyTorch reference."""
        eps = 1e-5
        device = "cuda"

        # Create data
        x_shape = (M, N)
        w_shape = (x_shape[-1],)
        weight = torch.rand(w_shape, dtype=dtype, device=device)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

        # Forward pass
        y_tri = rmsnorm_forward(x, weight, eps)
        y_ref = torch_rms_norm(x.to(torch.float32), weight.to(torch.float32), eps).to(dtype)

        # Compare
        max_delta = torch.max(torch.abs(y_tri - y_ref)).item()
        print(f"\ntest_rmsnorm: M={M}, N={N}, dtype={dtype}, max_delta={max_delta:.6f}")

        assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0), f"y mismatch! max_delta={max_delta}"


class TestAddRMSNormFused:
    """Test cases for Fused Add + RMSNorm kernel."""

    @pytest.mark.parametrize(
        "M, N, dtype",
        [
            (1024, 4096, torch.float16),
            (1024, 4096, torch.bfloat16),
            (512, 8192, torch.bfloat16),
            (1024, 16384, torch.bfloat16),
            (1024, 32760, torch.bfloat16),
        ],
    )
    def test_add_rmsnorm_fused_correctness(self, M, N, dtype):
        """Test Fused Add + RMSNorm output correctness against PyTorch reference."""
        eps = 1e-5
        device = "cuda"

        # Create data
        x_shape = (M, N)
        w_shape = (x_shape[-1],)
        weight = torch.rand(w_shape, dtype=dtype, device=device)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        residual = 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

        # Keep a copy of x for reference computation
        x_ref = x.clone()

        # Forward pass (note: x is modified in-place to x + residual)
        y_tri = add_rmsnorm_fused_forward(x, residual, weight, eps)

        # Compute reference:
        # 1. x_after = x + residual
        # 2. y = rmsnorm(x_after) * weight
        x_ref.add_(residual)
        y_ref = torch_rms_norm(x_ref.to(torch.float32), weight.to(torch.float32), eps).to(dtype)

        # Compare x (should be updated to x + residual)
        x_max_delta = torch.max(torch.abs(x - x_ref)).item()
        print(f"\ntest_add_rmsnorm_fused: M={M}, N={N}, dtype={dtype}")
        print(f"  x max_delta={x_max_delta:.6f}")
        assert torch.allclose(x, x_ref, atol=1e-2, rtol=0), f"x (residual update) mismatch! max_delta={x_max_delta}"

        # Compare y (normalized output)
        y_max_delta = torch.max(torch.abs(y_tri - y_ref)).item()
        print(f"  y max_delta={y_max_delta:.6f}")
        assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0), f"y mismatch! max_delta={y_max_delta}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_rmsnorm_fused_inplace_residual(self, dtype):
        """Test that input x is correctly updated in-place to x + residual."""
        M, N = 512, 4096
        eps = 1e-5
        device = "cuda"

        x = torch.randn(M, N, dtype=dtype, device=device)
        residual = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.rand(N, dtype=dtype, device=device)

        x_original = x.clone()
        expected_x = x_original + residual

        _ = add_rmsnorm_fused_forward(x, residual, weight, eps)

        # x should now be x_original + residual
        assert torch.allclose(x, expected_x, atol=1e-2, rtol=0), "x not updated correctly in-place"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_rmsnorm_fused_output_buffer(self, dtype):
        """Test that output buffer (out parameter) works correctly."""
        M, N = 256, 4096
        eps = 1e-5
        device = "cuda"

        x = torch.randn(M, N, dtype=dtype, device=device)
        residual = torch.randn(M, N, dtype=dtype, device=device)
        weight = torch.rand(N, dtype=dtype, device=device)
        out = torch.empty(M, N, dtype=dtype, device=device)

        x_copy = x.clone()
        y = add_rmsnorm_fused_forward(x, residual, weight, eps, out=out)

        # y should be the same as out
        assert y.data_ptr() == out.data_ptr(), "Output buffer not used correctly"

        # Verify correctness
        x_ref = x_copy + residual
        y_ref = torch_rms_norm(x_ref.to(torch.float32), weight.to(torch.float32), eps).to(dtype)
        assert torch.allclose(y, y_ref, atol=1e-2, rtol=0), "Output with buffer is incorrect"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
