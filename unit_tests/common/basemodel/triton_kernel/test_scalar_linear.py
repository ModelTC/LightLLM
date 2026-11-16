import pytest
import torch

from lightllm.common.basemodel.triton_kernel.scalar_linear import scalar_linear
from lightllm.common.quantization.scalar_projection import StableScalarProjection
from lightllm.common.quantization.quantize_method import WeightPack


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden", [65, 2560])
def test_reference_batching_strides_and_graph(dtype, hidden):
    torch.manual_seed(2319)
    input = torch.randn((336, hidden * 2), device="cuda", dtype=dtype)[1:335, ::2]
    weight = torch.randn((1, hidden * 2), device="cuda", dtype=dtype)[:, ::2]
    out = torch.empty((334, 3), device="cuda", dtype=dtype)[:, 1:2]
    expected = (input.double() * weight.double()).sum(-1, keepdim=True).to(dtype)
    assert scalar_linear(input, weight, out=out) is out
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    for size in (1, 3, 8, 24, 64, 128, 256, 334):
        pieces = [scalar_linear(chunk, weight) for chunk in input.split(size)]
        assert torch.equal(torch.cat(pieces), out)
    permutation = torch.randperm(334, device="cuda")
    assert torch.equal(scalar_linear(input[permutation], weight), out[permutation])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scalar_linear(input, weight, out=out)
    input.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    expected = (input.double() * weight.double()).sum(-1, keepdim=True).to(dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_small_terms_survive_before_final_dtype_cast():
    input = torch.tensor([[1, 2 ** -8, 2 ** -24, 2 ** -24]], device="cuda", dtype=torch.bfloat16)
    weight = torch.ones_like(input)
    actual = scalar_linear(input, weight)
    torch.testing.assert_close(actual, torch.full_like(actual, 1.0078125), atol=0, rtol=0)


def test_dispatch_preserves_weight_layout_output_and_fallbacks(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    method = StableScalarProjection()
    input = torch.randn((7, 65), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((1, 65), device="cuda", dtype=input.dtype)
    pack = WeightPack(weight=weight, weight_scale=None, weight_zero_point=None)
    out = torch.empty((7, 1), device="cuda", dtype=input.dtype)
    assert method.method_name == "none"
    assert method.apply(input, pack, out=out, use_custom_tensor_mananger=False) is out
    expected = (input.double() @ weight.double().t()).to(input.dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    torch.testing.assert_close(method.apply(input, pack), expected, atol=0, rtol=0)

    for outputs, bias in ((2, None), (1, torch.ones((1,), device="cuda", dtype=input.dtype))):
        weight = torch.randn((outputs, 65), device="cuda", dtype=input.dtype)
        pack = WeightPack(weight=weight, weight_scale=None, weight_zero_point=None)
        actual = method.apply(input, pack, bias=bias, use_custom_tensor_mananger=False)
        expected = torch.mm(input, weight.t()) if bias is None else torch.addmm(bias, input, weight.t())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_empty_input_and_invalid_output():
    input = torch.empty((0, 65), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((1, 65), device="cuda", dtype=input.dtype)
    assert scalar_linear(input, weight).shape == (0, 1)
    with pytest.raises(ValueError, match="output must match"):
        scalar_linear(input, weight, out=torch.empty((0, 2), device="cuda", dtype=input.dtype))
