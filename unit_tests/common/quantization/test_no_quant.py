import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.common.quantization.quantize_method import WeightPack


@pytest.fixture(autouse=True)
def device_env(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp32_output", [False, True])
def test_output_buffer_controls_matmul_precision(dtype, fp32_output):
    torch.manual_seed(1593)
    x = torch.randn((17, 64), dtype=dtype, device="cuda")
    weight = torch.randn((32, 64), dtype=dtype, device="cuda")
    out = torch.empty((17, 32), dtype=torch.float32 if fp32_output else dtype, device="cuda")

    projection = MMWeightTpl.__new__(MMWeightTpl)
    projection.quant_method = NoQuantization()
    projection.mm_param = WeightPack(weight=weight)
    projection.bias = None
    result = projection.mm(x, out=out)

    assert result is out
    expected = torch.mm(x.float(), weight.float().t()).to(out.dtype)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_default_output_and_bias_preserve_input_dtype():
    x = torch.randn((3, 64), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((32, 64), dtype=x.dtype, device=x.device)
    bias = torch.randn(32, dtype=x.dtype, device=x.device)

    result = NoQuantization().apply(x, WeightPack(weight=weight), bias=bias, use_custom_tensor_mananger=False)

    assert result.dtype == x.dtype
    torch.testing.assert_close(result, torch.addmm(bias, x, weight.t()), rtol=0, atol=0)
