import torch
import pytest
import torch.nn.functional as F
from lightllm.common.quantization.triton_quant.fp8.fp8w8a8_scaled_mm_per_token_kernel import fp8_scaled_mm_per_token


@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("N,K", [(2048, 2048), (4096, 5120), (8192, 4096)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16])
def test_fp8_scaled_mm_per_token_correctness(M, N, K, output_dtype):
    """Test the correctness of fp8_scaled_mm_per_token kernel.

    Args:
        M: Batch size / number of tokens
        N: Output dimension
        K: Hidden dimension
        output_dtype: Output data type (float16 or bfloat16)
    """
    # Prepare input matrices
    # A: [M, K] - activation matrix
    A = torch.randn((M, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)
    Ascale = torch.randn((M, 1)).cuda().abs() + 0.01  # Ensure positive scales

    # B: [K, N] - weight matrix
    B = torch.randn((N, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn).transpose(0, 1)
    Bscale = torch.randn((1, N)).cuda().abs() + 0.01  # Ensure positive scales

    # Prepare output matrix
    out = torch.zeros((M, N), dtype=output_dtype).cuda()

    # Compute ground truth using PyTorch
    d_A = A.to(output_dtype) * Ascale.to(output_dtype)
    d_B = B.to(output_dtype) * Bscale.to(output_dtype)
    gt_C = torch.mm(d_A, d_B)

    # Run the FP8 kernel
    result = fp8_scaled_mm_per_token(A, B, Ascale, Bscale, output_dtype, out)

    # Verify the result is the same as out (in-place operation)
    assert result is out

    # Check cosine similarity
    cosine_sim = F.cosine_similarity(out.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)

    # Check absolute errors
    max_abs_error = torch.max(torch.abs(out - gt_C)).item()
    mean_abs_error = torch.mean(torch.abs(out - gt_C)).item()

    # Print debug info on failure
    if cosine_sim.item() < 0.99:
        print(f"\n[FAILED] M={M}, N={N}, K={K}, dtype={output_dtype}")
        print(f"  Cosine Similarity: {cosine_sim.item():.6f}")
        print(f"  Max Absolute Error: {max_abs_error:.6e}")
        print(f"  Mean Absolute Error: {mean_abs_error:.6e}")

    # Assert correctness with a threshold
    assert cosine_sim.item() >= 0.99, (
        f"Cosine similarity {cosine_sim.item():.6f} < 0.99 " f"(M={M}, N={N}, K={K}, dtype={output_dtype})"
    )

    # Additional checks for numerical stability
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


@pytest.mark.parametrize("M,N,K", [(7, 1023, 2049)])
def test_fp8_scaled_mm_per_token_odd_shapes(M, N, K):
    """Test the kernel with odd/non-divisible shapes to verify masking logic."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    output_dtype = torch.bfloat16

    # Prepare input matrices with odd dimensions
    A = torch.randn((M, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn)
    Ascale = torch.randn((M, 1)).cuda().abs() + 0.01

    B = torch.randn((N, K), dtype=output_dtype).cuda().to(torch.float8_e4m3fn).transpose(0, 1)
    Bscale = torch.randn((1, N)).cuda().abs() + 0.01

    out = torch.zeros((M, N), dtype=output_dtype).cuda()

    # Compute ground truth
    d_A = A.to(output_dtype) * Ascale.to(output_dtype)
    d_B = B.to(output_dtype) * Bscale.to(output_dtype)
    gt_C = torch.mm(d_A, d_B)

    # Run the FP8 kernel
    fp8_scaled_mm_per_token(A, B, Ascale, Bscale, output_dtype, out)

    # Check cosine similarity
    cosine_sim = F.cosine_similarity(out.flatten().unsqueeze(0), gt_C.flatten().unsqueeze(0), dim=1)

    assert cosine_sim.item() >= 0.99, f"Cosine similarity {cosine_sim.item():.6f} < 0.99"


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke test...")
    print("Testing M=16, N=2048, K=4096, dtype=bfloat16")
    test_fp8_scaled_mm_per_token_correctness(M=128, N=812, K=4096, output_dtype=torch.bfloat16)
    print("✅ Smoke test passed!")

    print("\nRunning full test suite...")
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
