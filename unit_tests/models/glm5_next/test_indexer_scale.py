import pytest
import torch

from lightllm.models.glm5_next.triton_kernel.index_quant import hadamard_transform_quant_fp8


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("scale_format", ["auto", "fp32", "ue8m0"])
@pytest.mark.parametrize("rows", [0, 1, 33])
def test_indexer_scale_matches_hadamard_reference(monkeypatch, scale_format, rows):
    monkeypatch.setenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", scale_format)
    # FP32 matrix products are exact for these power-of-two input fractions.
    x = (torch.randint(-64, 64, (rows, 128), device="cuda").float() / 64).bfloat16()
    actual, scales = hadamard_transform_quant_fp8(x, 128 ** -0.5)
    assert actual.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert actual.shape == x.shape and scales.shape == (rows, 1)
    if not rows:
        return
    matrix = torch.ones((1, 1), device="cuda")
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    transformed = ((x.float() @ matrix) * 128 ** -0.5).bfloat16().float()
    expected_scale = transformed.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    if scale_format != "fp32":
        expected_scale = expected_scale.log2().ceil().exp2()
    torch.testing.assert_close(scales, expected_scale, rtol=1e-6, atol=0)
    expected = (transformed / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
