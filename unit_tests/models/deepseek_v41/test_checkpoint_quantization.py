import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.usefixtures("runtime_start_args"),
]


def test_v41_b128_checkpoint_uses_existing_fp8_quant_methods():
    from lightllm.common.quantization import Quantcfg
    from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel

    config = DeepseekV41TpPartModel.normalize_model_config(
        dict(
            model_type="deepseek_v41",
            text_config=dict(num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
            quantization_config=dict(
                quant_method="fp8", weight_block_size=[128, 128], expert_dtype="fp8", scale_fmt="ue8m0"
            ),
        )
    )
    quant = Quantcfg(config)
    assert quant.get_quant_method(0, "fused_moe").method_name == "fp8w8a8-b128-deepgemm"
    assert quant.get_quant_method(0, "wq_a").method_name == "fp8w8a8-b128-deepgemm"


def _assert_native_fp8_gemm_accuracy(actual, expected):
    # Bound its error against an FP64 product of the decoded quantized inputs.
    expected = expected.double()
    error = actual.double() - expected
    assert error.norm() <= 1e-4 * expected.norm()
    assert error.abs().max() <= 1e-3 * expected.square().mean().sqrt()


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("tokens", [1, 17, 256])
def test_block32_fp8_loader_and_native_gemm(scale_first, tokens):
    from lightllm.common.quantization.w8a8 import FP8w8a8B32TritonQuantizationMethod
    from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_gemm_kernel import (
        w8a8_block_fp8_matmul,
    )

    method = FP8w8a8B32TritonQuantizationMethod()
    method.hf_quantization_config = {"scale_fmt": "ue8m0"}
    layer = ROWMMWeight(128, [32, 32], ["gate.weight", "up.weight"], torch.bfloat16, quant_method=method)
    pack, parts = layer.mm_param, layer.mm_param_list
    raw = (torch.randn(64, 128) * 10).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-5, 0, (2, 4)).float()).to(torch.float8_e8m0fnu)
    for part, w, ws in zip(parts, raw.split(32), scale.split(1)):
        loaders = [(method.load_weight, w), (method.load_weight_scale, ws)]
        if scale_first:
            loaders.reverse()
        loaders[0][0](loaders[0][1], part)
        assert not all(part.load_ok)
        loaders[1][0](loaders[1][1], part)
        assert all(part.load_ok)
    assert pack.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(pack.weight.float().cpu(), raw.float(), atol=0, rtol=0)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    x[:, :32] = 0
    blocks = x.float().unflatten(-1, (4, 32))
    sx = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448)))
    qx = (blocks / sx[..., None]).to(torch.float8_e4m3fn).flatten(-2)
    dw = raw.float().cuda() * scale.float().cuda().repeat_interleave(32, 0).repeat_interleave(32, 1)
    dx = qx.float() * sx.repeat_interleave(32, -1)
    expected = (dx.double() @ dw.double().T).float()
    qa, _, sa, _, _ = method._dynamic_quant_input(x, pack, use_custom_tensor_mananger=False)
    torch.testing.assert_close(qa.float(), qx.float(), atol=0, rtol=0)
    torch.testing.assert_close(sa, sx, atol=0, rtol=0)
    out = torch.empty(tokens, 64, device="cuda", dtype=torch.float32)
    actual = method.apply(x, pack, out=out, use_custom_tensor_mananger=False)
    assert actual is out
    _assert_native_fp8_gemm_accuracy(actual, expected)
    # Autotune candidates with K=64/128 must still respect the checkpoint's K=32 scales.
    for tile_k in (32, 64, 128):
        w8a8_block_fp8_matmul(
            qx,
            pack.weight.T,
            sx,
            pack.weight_scale.T,
            out,
            (32, 32),
            run_config=dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=tile_k, GROUP_M=8, num_warps=4, num_stages=3),
        )
        _assert_native_fp8_gemm_accuracy(out, expected)
