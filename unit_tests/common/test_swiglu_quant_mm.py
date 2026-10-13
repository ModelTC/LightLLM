import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import moe_silu_and_mul as activation
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_quant_kernel import weight_quant
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl
from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.common.quantization.quantize_method import WeightPack


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def unfused_quant(input, limit):
    output = torch.empty((input.shape[0], input.shape[1] // 2), device=input.device, dtype=input.dtype)
    activation.silu_and_mul_fwd(input, output, limit=limit)
    return per_token_group_quant_fp8(
        output, 128, column_major_scales=True, scale_tma_aligned=True, use_ue8m0_scales=True
    )


@pytest.mark.parametrize("m,n", [(1, 128), (7, 256), (64, 2048), (256, 2048)])
@pytest.mark.parametrize("limit", [None, 7.0])
@pytest.mark.parametrize("gelu", [False, True])
def test_activation_quant_preserves_bytes_scales_and_existing_activation_selector(monkeypatch, m, n, limit, gelu):
    monkeypatch.setattr(activation, "ffn_use_tanh_approximate_gelu", lambda: gelu)
    torch.manual_seed(1013)
    input = torch.randn(m, 2 * n, dtype=torch.bfloat16, device="cuda") * 10
    input[:, :128] = 1e-7
    input[0].zero_()
    expected_q, expected_s = unfused_quant(input, limit)
    q, s = activation.silu_and_mul_quant_fp8(input, 128, torch.float8_e4m3fn, limit=limit)
    torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
    assert q.is_contiguous() and s.stride() == (1, ((m + 3) // 4) * 4)


def test_fused_activation_graph_replay():
    input = torch.zeros(7, 512, dtype=torch.bfloat16, device="cuda")
    activation.silu_and_mul_quant_fp8(input, 128, torch.float8_e4m3fn, limit=7.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = activation.silu_and_mul_quant_fp8(input, 128, torch.float8_e4m3fn, limit=7.0)
    for factor in [1.0, 10.0]:
        input.normal_().mul_(factor)
        graph.replay()
        expected_q, expected_s = unfused_quant(input, 7.0)
        torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)


@pytest.mark.parametrize("m", [7, 64, 256])
def test_selected_deepgemm_weight_gives_identical_real_w2_output(m):
    torch.manual_seed(1013)
    gate_up = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda") * 10
    weight = torch.randn(4096, 2048, dtype=torch.bfloat16, device="cuda")
    q, s = weight_quant(weight, 128, use_ue8m0_scales=True)
    owner = MMWeightTpl.__new__(MMWeightTpl)
    owner.quant_method = DeepGEMMFP8w8a8B128QuantizationMethod()
    owner.mm_param = WeightPack(weight=q, weight_scale=s)
    owner.bias = None
    activated = torch.empty((m, 2048), dtype=gate_up.dtype, device=gate_up.device)
    activation.silu_and_mul_fwd(gate_up, activated, limit=7.0)
    expected = owner.mm(activated, use_custom_tensor_mananger=False)
    actual = owner.mm_swiglu(gate_up, limit=7.0, use_custom_tensor_mananger=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selected_no_quant_weight_preserves_activation_and_bias():
    gate_up = torch.randn(7, 512, dtype=torch.bfloat16, device="cuda") * 10
    owner = MMWeightTpl.__new__(MMWeightTpl)
    owner.quant_method = NoQuantization()
    owner.mm_param = WeightPack(weight=torch.randn(128, 256, dtype=torch.bfloat16, device="cuda"))
    owner.bias = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    activated = torch.empty((7, 256), dtype=gate_up.dtype, device=gate_up.device)
    activation.silu_and_mul_fwd(gate_up, activated, limit=7.0)
    expected = owner.mm(activated, use_custom_tensor_mananger=False)
    output = torch.empty_like(expected)
    actual = owner.mm_swiglu(gate_up, limit=7.0, out=output, use_custom_tensor_mananger=False)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
