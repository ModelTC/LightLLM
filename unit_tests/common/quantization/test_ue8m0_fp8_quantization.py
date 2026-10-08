from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.quantization import fp8act_quant_kernel as activation
from lightllm.common.basemodel.triton_kernel.quantization import fp8w8a8_block_quant_kernel as weight
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization import deepgemm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@triton.jit
def _round_scales_kernel(x, act_out, weight_out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets, mask=offsets < N, other=1.0)
    tl.store(act_out + offsets, activation._ceil_to_ue8m0(values), mask=offsets < N)
    tl.store(weight_out + offsets, weight._ceil_to_ue8m0(values), mask=offsets < N)


def test_ue8m0_rounding_at_power_of_two_boundaries():
    powers = torch.tensor([2.0 ** exponent for exponent in (-126, -24, -8, 0, 8, 126, 127)], device="cuda")
    above = torch.nextafter(powers, torch.full_like(powers, float("inf")))
    below = torch.nextafter(powers, torch.zeros_like(powers))
    scales = torch.cat((powers, above, below, powers.new_zeros(1)))
    expected = torch.cat((powers, (powers * 2).clamp_max(2.0 ** 127), powers, powers.new_tensor([2.0 ** -126])))
    act_out = torch.empty_like(scales)
    weight_out = torch.empty_like(scales)

    _round_scales_kernel[(1,)](scales, act_out, weight_out, scales.numel(), triton.next_power_of_2(scales.numel()))

    torch.testing.assert_close(act_out, expected, rtol=0, atol=0)
    torch.testing.assert_close(weight_out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["row", "column", "tma"])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("use_ue8m0_scales", [False, True])
def test_activation_scales_and_quantized_values(monkeypatch, layout, group_size, use_ue8m0_scales):
    monkeypatch.setattr(activation, "HAS_SGL_KERNEL", False)
    torch.manual_seed(20261008)
    rows, groups = 17, 4
    x = torch.randn(rows, groups * group_size, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1].fill_(1e-12)
    q, scales = activation.per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=layout != "row",
        scale_tma_aligned=layout == "tma",
        use_ue8m0_scales=use_ue8m0_scales,
    )

    amax = x.float().reshape(rows, groups, group_size).abs().amax(dim=-1)
    reference_scales = amax.clamp_min(1e-4 if use_ue8m0_scales else 1e-10) / 448.0
    if use_ue8m0_scales:
        reference_scales = torch.exp2(torch.ceil(torch.log2(reference_scales.double()))).float()
    reference_q = (x.float().reshape(rows, groups, group_size) / reference_scales[..., None]).to(q.dtype)
    torch.testing.assert_close(scales, reference_scales, rtol=0, atol=0)
    torch.testing.assert_close(q.float(), reference_q.reshape_as(x).float(), rtol=0, atol=0)
    # The existing LightLLM fallback converts scales only when TMA alignment is requested.
    assert scales.stride() == ((1, 20) if layout == "tma" else (groups, 1))


def test_ue8m0_bypasses_sgl_kernel(monkeypatch):
    monkeypatch.setattr(activation, "HAS_SGL_KERNEL", True)

    def unexpected_sgl_call(*args, **kwargs):
        pytest.fail("UE8M0 quantization must use the kernel that rounds scales")

    monkeypatch.setattr(activation, "sgl_ops", SimpleNamespace(sgl_per_token_group_quant_fp8=unexpected_sgl_call))
    x = torch.full((3, 128), 1.0, device="cuda", dtype=torch.bfloat16)
    _, scales = activation.per_token_group_quant_fp8(x, 128, use_ue8m0_scales=True)
    torch.testing.assert_close(scales, torch.full_like(scales, 2.0 ** -8), rtol=0, atol=0)


@pytest.mark.parametrize("experts", [None, 2])
@pytest.mark.parametrize("use_ue8m0_scales", [False, True])
def test_weight_quantization_partial_blocks(monkeypatch, experts, use_ue8m0_scales):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    torch.manual_seed(20261008)
    shape = (129, 257) if experts is None else (experts, 129, 257)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x[..., :128, :128].zero_()
    x[..., 128:, 128:256].fill_(1e-12)
    q, scales = weight.weight_quant(x, use_ue8m0_scales=use_ue8m0_scales)

    blocks = F.pad(x.float(), (0, 127, 0, 127)).reshape(-1, 2, 128, 3, 128)
    amax = blocks.abs().amax(dim=(2, 4)).reshape_as(scales)
    reference_scales = amax.clamp_min(1e-4) / 448.0 if use_ue8m0_scales else amax / 448.0
    if use_ue8m0_scales:
        reference_scales = torch.exp2(torch.ceil(torch.log2(reference_scales.double()))).float()
    denom = reference_scales if use_ue8m0_scales else reference_scales + 1e-6
    expanded = denom.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)[..., :129, :257]
    torch.testing.assert_close(scales, reference_scales, rtol=0, atol=0)
    torch.testing.assert_close(q.float(), (x.float() / expanded).to(q.dtype).float(), rtol=0, atol=0)


@pytest.mark.skipif(not deepgemm.HAS_DEEPGEMM, reason="requires DeepGEMM")
@pytest.mark.parametrize("scale_fmt", ["no_config", None, "ue8m0", "float32"])
@pytest.mark.parametrize("rows", [1, 17, 32])
def test_deepgemm_quantize_and_apply(monkeypatch, rows, scale_fmt):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    torch.manual_seed(20261008)
    x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    config = {"n_layer": 1}
    if scale_fmt != "no_config":
        config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        if scale_fmt is not None:
            config["quantization_config"]["scale_fmt"] = scale_fmt
    method = Quantcfg(config, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    use_ue8m0_scales = scale_fmt != "float32"
    assert method.use_ue8m0_scales == use_ue8m0_scales
    weight_pack, _ = method.create_weight([256], 1024, torch.bfloat16, 0)
    method.quantize(w, weight_pack)
    assert torch.all(weight_pack.weight_scale > 0)
    log_scales = torch.log2(weight_pack.weight_scale)
    assert torch.equal(log_scales, log_scales.round()) == use_ue8m0_scales

    quantize_activation = deepgemm.per_token_group_quant_fp8

    def check_activation_scales(*args, **kwargs):
        result = quantize_activation(*args, **kwargs)
        log_scales = torch.log2(result[1])
        assert torch.equal(log_scales, log_scales.round()) == use_ue8m0_scales
        return result

    monkeypatch.setattr(deepgemm, "per_token_group_quant_fp8", check_activation_scales)

    out = method.apply(x, weight_pack, use_custom_tensor_mananger=False)
    reference = x.float() @ w.float().T
    nrmse = (out.float() - reference).square().mean().sqrt() / reference.square().mean().sqrt()
    cosine = F.cosine_similarity(out.float().flatten(), reference.flatten(), dim=0)
    assert torch.isfinite(out).all()
    assert nrmse.item() < 0.08
    assert cosine.item() > 0.995
