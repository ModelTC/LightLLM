import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import (
    moe_silu_and_mul as activation,
)
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
)
from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_quant_kernel import (
    weight_quant,
)
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeightTpl,
)
from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.common.quantization.quantize_method import WeightPack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def unfused(input, limit, alpha, add_one):
    out = torch.empty((input.shape[0], input.shape[1] // 2), device=input.device, dtype=input.dtype)
    activation.silu_and_mul_fwd(input, out, limit=limit, alpha=alpha, clamp_up_add_one=add_one)
    return per_token_group_quant_fp8(
        out,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        use_ue8m0_scales=True,
    )


@pytest.mark.parametrize("m,n", [(1, 128), (7, 256), (33, 2048), (128, 2048), (256, 2048)])
@pytest.mark.parametrize("limit,alpha,add_one", [(None, None, True), (7.0, 1.0, False), (7.0, 1.7, True)])
def test_fused_preserves_activation_rounding_and_ue8m0_bytes(m, n, limit, alpha, add_one):
    torch.manual_seed(1013)
    input = torch.randn(m, 2 * n, dtype=torch.bfloat16, device="cuda") * 10
    input[:, :128] = 1e-7
    input[0].zero_()
    expected_q, expected_s = unfused(input, limit, alpha, add_one)
    q, s = activation.silu_and_mul_quant_fp8(
        input,
        128,
        torch.float8_e4m3fn,
        limit=limit,
        alpha=alpha,
        clamp_up_add_one=add_one,
    )
    torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
    assert q.is_contiguous() and s.stride() == (1, ((m + 3) // 4) * 4)


def test_glm_fused_graph_replay():
    input = torch.zeros(7, 512, dtype=torch.bfloat16, device="cuda")
    kwargs = dict(limit=7.0, alpha=1.0, clamp_up_add_one=False)
    activation.silu_and_mul_quant_fp8(input, 128, torch.float8_e4m3fn, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = activation.silu_and_mul_quant_fp8(input, 128, torch.float8_e4m3fn, **kwargs)
    for factor in [1.0, 10.0]:
        input.normal_().mul_(factor)
        graph.replay()
        expected_q, expected_s = unfused(input, 7.0, 1.0, False)
        torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)


def test_glm_fused_reuses_kernel_with_varying_scale_strides(monkeypatch):
    kernel = activation._silu_and_mul_quant_fp8_kernel
    original = kernel.run
    hashes = []

    def run(*args, **kwargs):
        compiled = original(*args, **kwargs)
        hashes.append(compiled.hash)
        return compiled

    monkeypatch.setattr(kernel, "run", run)
    for m in [33, 64, 65, 257]:
        input = torch.randn(m, 512, dtype=torch.bfloat16, device="cuda")
        actual = activation.silu_and_mul_quant_fp8(
            input,
            128,
            torch.float8_e4m3fn,
            limit=7.0,
            alpha=1.0,
            clamp_up_add_one=False,
        )
        expected = unfused(input, 7.0, 1.0, False)
        torch.testing.assert_close(actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert len(hashes) == 4 and len(set(hashes)) == 1


@pytest.mark.parametrize("m", [7, 64, 256])
def test_glm_fused_deepgemm_w2_output(m):
    torch.manual_seed(1013)
    input = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda") * 10
    weight = torch.randn(4096, 2048, dtype=torch.bfloat16, device="cuda")
    qw, sw = weight_quant(weight, 128, use_ue8m0_scales=True)
    owner = MMWeightTpl.__new__(MMWeightTpl)
    owner.quant_method = DeepGEMMFP8w8a8B128QuantizationMethod()
    owner.quant_method.use_ue8m0_scales = True
    owner.quant_method.use_packed_ue8m0 = False
    owner.mm_param = WeightPack(weight=qw, weight_scale=sw)
    owner.bias = None
    activated = torch.empty((m, 2048), dtype=input.dtype, device=input.device)
    activation.silu_and_mul_fwd(input, activated, limit=7.0, alpha=1.0, clamp_up_add_one=False)
    expected = owner.mm(activated, use_custom_tensor_mananger=False)
    actual = owner.mm_swiglu(
        input,
        limit=7.0,
        alpha=1.0,
        clamp_up_add_one=False,
        use_custom_tensor_mananger=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_glm_fallback_preserves_no_quant_bias_and_output_buffer():
    input = torch.randn(7, 512, dtype=torch.bfloat16, device="cuda") * 10
    owner = MMWeightTpl.__new__(MMWeightTpl)
    owner.quant_method = NoQuantization()
    owner.mm_param = WeightPack(weight=torch.randn(128, 256, dtype=input.dtype, device=input.device))
    owner.bias = torch.randn(128, dtype=input.dtype, device=input.device)
    activated = torch.empty((7, 256), dtype=input.dtype, device=input.device)
    activation.silu_and_mul_fwd(input, activated, limit=7.0, alpha=1.0, clamp_up_add_one=False)
    expected = owner.mm(activated, use_custom_tensor_mananger=False)
    output = torch.empty_like(expected)
    actual = owner.mm_swiglu(
        input,
        limit=7.0,
        alpha=1.0,
        clamp_up_add_one=False,
        out=output,
        use_custom_tensor_mananger=False,
    )
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
