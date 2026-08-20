import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward, torch_rms_norm


@pytest.mark.parametrize("M,N", [(64, 256), (17, 1024), (128, 128)])
@pytest.mark.parametrize("has_weight", [True, False])
def test_rmsnorm_contiguous(M, N, has_weight):
    """Ordinary contiguous input: the shape the model paths use today."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for rmsnorm test")

    torch.manual_seed(0)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)
    weight = torch.rand((N,), device="cuda", dtype=torch.float32) if has_weight else None

    out = rmsnorm_forward(x, weight, eps=1e-6)
    ref = torch_rms_norm(x, weight if weight is not None else 1.0, 1e-6)

    assert (out - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize("M,N", [(64, 256), (17, 1024), (128, 128)])
@pytest.mark.parametrize("has_weight", [True, False])
def test_rmsnorm_last_dim_stride(M, N, has_weight):
    """The kernel takes x_stride1, so an input whose last-dim stride is not 1 must work.

    `rmsnorm_forward` reaches the kernel with one without complaint: `x.view(-1, N)` is
    shape-preserving for a 2-D input, and `torch.empty_like` gives the output the same
    strides.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for rmsnorm test")

    torch.manual_seed(0)
    x = torch.randn((N, M), device="cuda", dtype=torch.float32).t()
    assert x.shape == (M, N) and x.stride(1) != 1
    weight = torch.rand((N,), device="cuda", dtype=torch.float32) if has_weight else None

    out = rmsnorm_forward(x, weight, eps=1e-6)
    ref = torch_rms_norm(x, weight if weight is not None else 1.0, 1e-6)

    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-5, f"max diff too large: {max_diff}"

    # and it must agree with the same values laid out contiguously
    contiguous_out = rmsnorm_forward(x.contiguous(), weight, eps=1e-6)
    assert (out - contiguous_out).abs().max().item() < 1e-5
