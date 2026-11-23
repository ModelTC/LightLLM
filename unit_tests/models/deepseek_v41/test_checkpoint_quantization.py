from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.usefixtures("runtime_start_args"),
]


@pytest.mark.parametrize("expert_dtype", [None, "fp4"])
def test_v41_mxfp4_uses_quant_selector_on_hopper(expert_dtype):
    from lightllm.common.quantization import Quantcfg

    config = dict(
        model_type="deepseek_v41",
        n_layer=2,
        expert_dtype="fp4",
        quantization_config=dict(quant_method="fp8", weight_block_size=[32, 32]),
    )
    quant = Quantcfg(config, expert_dtype=expert_dtype)
    assert quant.get_quant_method(0, "fused_moe").method_name == "mxfp4w4a16-b32-marlin"
    assert quant.get_quant_method(0, "wq_a").method_name == "fp8w8a8-b32-triton"


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


@pytest.mark.parametrize("intermediate_size", [128, 288, 576])
def test_hopper_mxfp4_marlin_padding_preserves_clamped_experts(intermediate_size):
    from lightllm.common.quantization.deepgemm import MXFP4MoEQuantizationMethod
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.marlin_impl import FuseMoeMarlin

    method = MXFP4MoEQuantizationMethod()
    w13, parts = method.create_moe_weight([intermediate_size, intermediate_size], 128, torch.bfloat16, 0, num_experts=2)
    w2, _ = method.create_moe_weight([128], intermediate_size, torch.bfloat16, 0, num_experts=2)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    raw, decoded = [], []
    for n, k in [(intermediate_size, 128), (intermediate_size, 128), (128, intermediate_size)]:
        codes = torch.randint(0, 16, (2, n, k))
        raw.append((codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.int8))
        decoded.append((table[codes] * 0.03125).bfloat16().cuda())
    for e in range(2):
        for pack, source in [
            (parts[0].get_expert(e), raw[0][e]),
            (parts[1].get_expert(e), raw[1][e]),
            (w2.get_expert(e), raw[2][e]),
        ]:
            method.load_weight(source, pack)
            method.load_weight_scale(
                torch.full((source.shape[0], source.shape[1] // 16), 0.03125).to(torch.float8_e8m0fnu), pack
            )
    holder = SimpleNamespace(w13=w13, w2=w2, data_type_=torch.bfloat16)
    method.finalize_moe_weight(holder)
    expert = FuseMoeMarlin.__new__(FuseMoeMarlin)
    expert.quant_method = method
    expert.n_routed_experts = 2
    expert.workspace = expert.create_workspace()
    x = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda") * 2
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]], device="cuda")
    weights = torch.tensor([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]], device="cuda")
    actual = expert._fused_experts(
        x, w13, w2, weights, ids, is_prefill=True, alpha=1.0, limit=1.0, clamp_up_add_one=False
    )
    expected = torch.zeros_like(x, dtype=torch.float32)
    for row in range(3):
        for i in range(2):
            e = ids[row, i]
            gate = torch.nn.functional.linear(x[row], decoded[0][e]).clamp(max=1.0)
            up = torch.nn.functional.linear(x[row], decoded[1][e]).clamp(-1.0, 1.0)
            hidden = torch.nn.functional.silu(gate.float()).bfloat16() * up
            expected[row] += torch.nn.functional.linear(hidden, decoded[2][e]).float() * weights[row, i]
    torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.03)
