import torch
import pytest

from lightllm.models.qwen2_vl.triton_kernel.rotary_pos_emb import apply_rotary_pos_emb_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


@pytest.mark.parametrize(
    "shape",
    [
        (1296, 64, 80),
        (1024, 2, 192),
        (1024, 1, 256),
        (1024, 3, 160),
    ],
)
def test_triton_matches_reference(shape):
    L, H, D = shape
    assert D % 2 == 0

    torch.manual_seed(0)

    freqs = torch.randn(L, D // 2, device="cuda", dtype=torch.bfloat16)
    cos = freqs.cos()
    sin = freqs.sin()

    tensor = torch.randn(L, H, D, device="cuda", dtype=torch.bfloat16)

    ref = apply_rotary_pos_emb_vision(tensor.unsqueeze(0), cos, sin).squeeze(0)
    out = apply_rotary_pos_emb_triton(tensor, cos, sin)

    assert out.dtype == tensor.dtype, "输出 dtype 应与输入一致"
    assert out.shape == tensor.shape, "输出形状应与输入一致"
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), "Triton 与参考实现不一致"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strided_qkv_view_uses_output_strides(dtype):
    torch.manual_seed(1907)
    qkv = torch.randn(37, 3, 3, 72, device="cuda", dtype=dtype)
    tensor = qkv[:, 1]
    assert not tensor.is_contiguous()
    original = qkv.clone()
    angles = torch.randn(37, 36, device="cuda", dtype=torch.float32)
    cos, sin = angles.cos(), angles.sin()
    expected = apply_rotary_pos_emb_vision(tensor.unsqueeze(0), cos, sin).squeeze(0)
    actual = apply_rotary_pos_emb_triton(tensor, cos, sin)
    tolerance = 0.016 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(qkv, original, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main()
