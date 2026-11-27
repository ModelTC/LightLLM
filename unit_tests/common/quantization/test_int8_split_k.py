import pytest
import torch

import lightllm.common.basemodel
from lightllm.common.quantization import w8a8
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils.device_utils import is_hopper, triton_support_tensor_descriptor


@pytest.mark.parametrize(
    "rows,width,depth,dtype",
    [(1, 4096, 4096, torch.bfloat16), (16, 3072, 7680, torch.float16), (512, 4096, 4096, torch.bfloat16)],
)
def test_int8_triton_out(monkeypatch, rows, width, depth, dtype):
    if not torch.cuda.is_available() or not is_hopper():
        pytest.skip("requires Hopper")
    if rows == 512 and not triton_support_tensor_descriptor():
        pytest.skip("requires TMA")
    torch.manual_seed(1767)
    inputs = torch.randn((rows, depth), device="cuda", dtype=dtype)
    weight = torch.randint(-127, 128, (width, depth), device="cuda", dtype=torch.int8)
    scales = torch.rand((width,), device="cuda", dtype=torch.float32) / 127
    pack = WeightPack(weight=weight, weight_scale=scales)
    output = torch.empty((rows, width), device="cuda", dtype=dtype)
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    method = w8a8.w8a8QuantizationMethod()
    monkeypatch.setattr(w8a8, "_sgl_scaled_mm", lambda *args: pytest.fail("expected Triton"))

    assert method.apply(inputs, pack, out=output) is output
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        method.apply(inputs, pack, out=output)
    inputs.mul_(-1)
    graph.replay()

    quantized, input_scales = w8a8.quantize_int8_per_row(inputs)
    if rows == 512:
        acc = torch._int_mm(quantized, weight.t())
    else:
        acc = quantized.cpu().int() @ weight.cpu().int().t()
    expected = (acc.cuda().float() * input_scales * scales).to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
