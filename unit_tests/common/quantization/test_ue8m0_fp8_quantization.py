import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.quantization import fp8act_quant_kernel as activation
from lightllm.common.basemodel.triton_kernel.quantization import fp8w8a8_block_quant_kernel as weight
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization import deepgemm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _unpack_scales(packed, groups):
    shifts = torch.arange(4, device=packed.device) * 8
    exponents = ((packed.to(torch.int64)[..., None] >> shifts) & 255).flatten(1)
    return torch.exp2(exponents[:, :groups].float() - 127)


@pytest.mark.skipif(not activation.HAS_SGL_KERNEL, reason="requires SGL kernel")
@pytest.mark.parametrize("use_ue8m0_scales,use_packed_ue8m0", [(False, False), (True, False), (True, True)])
def test_sgl_dispatch_uses_ue8m0_flag(monkeypatch, use_ue8m0_scales, use_packed_ue8m0):
    quantize = activation.sgl_ops.sgl_per_token_group_quant_fp8
    calls = []

    def record_call(*args, **kwargs):
        calls.append(kwargs["scale_ue8m0"])
        return quantize(*args, **kwargs)

    monkeypatch.setattr(activation.sgl_ops, "sgl_per_token_group_quant_fp8", record_call)
    x = torch.ones((3, 512), device="cuda", dtype=torch.bfloat16)
    _, scales = activation.per_token_group_quant_fp8(
        x, 128, use_ue8m0_scales=use_ue8m0_scales, use_packed_ue8m0=use_packed_ue8m0
    )
    assert calls == ([use_packed_ue8m0] if not use_ue8m0_scales or use_packed_ue8m0 else [])
    assert scales.dtype == (torch.int32 if use_packed_ue8m0 else torch.float32)
    if use_ue8m0_scales:
        actual = _unpack_scales(scales, 4) if use_packed_ue8m0 else scales
        torch.testing.assert_close(actual, torch.full_like(actual, 2.0 ** -8), rtol=0, atol=0)


@pytest.mark.parametrize("use_packed_ue8m0", [False, True])
def test_ue8m0_rounding_at_power_of_two_boundaries(monkeypatch, use_packed_ue8m0):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setattr(activation, "HAS_SGL_KERNEL", False)
    powers = torch.tensor([2.0 ** exponent for exponent in (-16, -8, 0, 8, 16, 118)], device="cuda")
    above = torch.nextafter(powers, torch.full_like(powers, float("inf")))
    below = torch.nextafter(powers, torch.zeros_like(powers))
    magnitudes = torch.cat((powers, above, below, powers.new_zeros(1))) * 448.0
    x = magnitudes[:, None].expand(-1, 128).contiguous()
    expected_act = torch.exp2(torch.ceil(torch.log2(magnitudes.clamp_min(1e-10) / 448.0)))
    expected_weight = torch.exp2(torch.ceil(torch.log2(magnitudes.clamp_min(1e-4) / 448.0)))

    _, act_scales = activation.per_token_group_quant_fp8(
        x, 128, use_ue8m0_scales=True, use_packed_ue8m0=use_packed_ue8m0
    )
    _, weight_scales = weight.weight_quant(x.repeat_interleave(128, dim=0), use_ue8m0_scales=True)

    actual = _unpack_scales(act_scales, 1) if use_packed_ue8m0 else act_scales
    torch.testing.assert_close(actual[:, 0], expected_act, rtol=0, atol=0)
    torch.testing.assert_close(weight_scales[:, 0], expected_weight, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["row", "column", "tma"])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("use_ue8m0_scales,use_packed_ue8m0", [(False, False), (True, False), (True, True)])
def test_activation_scales_and_quantized_values(monkeypatch, layout, group_size, use_ue8m0_scales, use_packed_ue8m0):
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
        use_packed_ue8m0=use_packed_ue8m0,
    )

    amax = x.float().reshape(rows, groups, group_size).abs().amax(dim=-1)
    reference_scales = amax.clamp_min(1e-10) / 448.0
    if use_ue8m0_scales:
        reference_scales = torch.exp2(torch.ceil(torch.log2(reference_scales.double()))).float()
    reference_q = (x.float().reshape(rows, groups, group_size) / reference_scales[..., None]).to(q.dtype)
    actual_scales = _unpack_scales(scales, groups) if use_packed_ue8m0 else scales
    torch.testing.assert_close(actual_scales, reference_scales, rtol=0, atol=0)
    torch.testing.assert_close(q.float(), reference_q.reshape_as(x).float(), rtol=0, atol=0)
    assert scales.dtype == (torch.int32 if use_packed_ue8m0 else torch.float32)
    # UE8M0 always follows SGL's packed TMA layout, including with default layout flags.
    if use_packed_ue8m0 or layout == "tma":
        assert scales.stride() == (1, 20)
    elif layout == "row":
        assert scales.stride() == (groups, 1)


@pytest.mark.parametrize("use_sgl", [False, True])
@pytest.mark.parametrize("rows", [1, 17])
@pytest.mark.parametrize("groups", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("group_size", [64, 128])
def test_packed_ue8m0_scales(monkeypatch, rows, groups, group_size, use_sgl):
    if use_sgl and not activation.HAS_SGL_KERNEL:
        pytest.skip("requires SGL kernel")
    monkeypatch.setattr(activation, "HAS_SGL_KERNEL", use_sgl)
    torch.manual_seed(20261008)
    x = torch.randn(rows, groups * group_size, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    if rows > 1:
        x[1].fill_(1e-12)
        x[-1, -group_size:].fill_(2.0 ** 16)
    q, packed_scales = activation.per_token_group_quant_fp8(x, group_size, use_ue8m0_scales=True, use_packed_ue8m0=True)

    assert packed_scales.dtype == torch.int32
    assert packed_scales.shape == (rows, (groups + 3) // 4)
    assert packed_scales.stride() == (1, (rows + 3) // 4 * 4)
    if groups % 4:
        assert torch.count_nonzero(packed_scales[:, -1].to(torch.int64) >> (groups % 4 * 8)) == 0
    scales = _unpack_scales(packed_scales, groups)
    amax = x.float().reshape(rows, groups, group_size).abs().amax(-1)
    reference_scales = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-10) / 448.0)))
    reference_q = (x.float().reshape(rows, groups, group_size) / reference_scales[..., None]).to(q.dtype)
    torch.testing.assert_close(scales, reference_scales, rtol=0, atol=0)
    torch.testing.assert_close(q.float(), reference_q.reshape_as(x).float(), rtol=0, atol=0)


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


@pytest.mark.parametrize("env_value", ["1", "0"])
def test_ue8m0_env_preserves_unquantized_method(monkeypatch, env_value):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", env_value)
    method = Quantcfg({"n_layer": 1}).get_quant_method(0, "q_proj")
    assert method.method_name == "none"


@pytest.mark.skipif(not deepgemm.HAS_DEEPGEMM, reason="requires DeepGEMM")
@pytest.mark.parametrize("sm100", [False, True])
def test_deepgemm_selects_scale_format_for_machine(monkeypatch, sm100):
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    monkeypatch.setattr(deepgemm, "is_sm100_gpu", lambda: sm100)
    method = Quantcfg({"n_layer": 1}, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    weight_pack, _ = method.create_weight([128], 512, torch.bfloat16, 0)
    method.load_weight(torch.ones((128, 512), dtype=torch.bfloat16, device="cuda"), weight_pack)
    seen = []

    def check_gemm_inputs(a, b, out):
        seen.append(a[1].dtype)
        scales = _unpack_scales(a[1], 4) if sm100 else a[1]
        torch.testing.assert_close(scales, torch.full_like(scales, 2.0 ** -8), rtol=0, atol=0)
        out.zero_()

    # Exercise format selection and real quantization on Hopper; SM100 GEMM needs a separate machine.
    monkeypatch.setattr(deepgemm, "_deepgemm_fp8_nt", check_gemm_inputs)
    method.apply(
        torch.ones((3, 512), dtype=torch.bfloat16, device="cuda"), weight_pack, use_custom_tensor_mananger=False
    )
    assert seen == [torch.int32 if sm100 else torch.float32]


@pytest.mark.skipif(not deepgemm.HAS_DEEPGEMM, reason="requires DeepGEMM")
@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("env_value", [None, "1", "0"])
@pytest.mark.parametrize("scale_fmt", ["no_config", None, "ue8m0", "float32"])
@pytest.mark.parametrize("rows", [1, 17, 32])
def test_deepgemm_load_and_apply(monkeypatch, rows, scale_fmt, env_value, prequantized):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    if env_value is None:
        monkeypatch.delenv("LIGHTLLM_USE_UE8M0_SCALES", raising=False)
    else:
        monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", env_value)
    torch.manual_seed(20261008)
    x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    config = {"n_layer": 1}
    if scale_fmt != "no_config":
        config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        if scale_fmt is not None:
            config["quantization_config"]["scale_fmt"] = scale_fmt
    method = Quantcfg(config, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    use_ue8m0_scales = scale_fmt == "ue8m0" or env_value == "1"
    weight_pack, _ = method.create_weight([256], 1024, torch.bfloat16, 0)
    assert method.use_ue8m0_scales == use_ue8m0_scales
    assert method.use_packed_ue8m0 == (use_ue8m0_scales and deepgemm.is_sm100_gpu())
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "0" if use_ue8m0_scales else "1")
    if prequantized:
        qweight, scales = weight.weight_quant(w, use_ue8m0_scales=use_ue8m0_scales)
        method.load_weight(qweight, weight_pack)
        method.load_weight_scale(scales, weight_pack)
    else:
        method.load_weight(w, weight_pack)
        assert method.use_ue8m0_scales == use_ue8m0_scales
    assert torch.all(weight_pack.weight_scale > 0)
    log_scales = torch.log2(weight_pack.weight_scale)
    assert torch.equal(log_scales, log_scales.round()) == use_ue8m0_scales

    quantize_activation = deepgemm.per_token_group_quant_fp8

    def check_activation_scales(*args, **kwargs):
        result = quantize_activation(*args, **kwargs)
        packed = use_ue8m0_scales and deepgemm.is_sm100_gpu()
        assert kwargs["use_packed_ue8m0"] == packed
        assert result[1].dtype == (torch.int32 if packed else torch.float32)
        if not packed:
            log_scales = torch.log2(result[1])
            assert torch.equal(log_scales, log_scales.round()) == use_ue8m0_scales
        return result

    monkeypatch.setattr(deepgemm, "per_token_group_quant_fp8", check_activation_scales)

    out = method.apply(x, weight_pack, use_custom_tensor_mananger=False)
    assert method.use_ue8m0_scales == use_ue8m0_scales
    # Later calls reuse the choice made when creating the weight.
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "0" if use_ue8m0_scales else "1")
    method.apply(x, weight_pack, out=out, use_custom_tensor_mananger=False)
    reference = x.float() @ w.float().T
    nrmse = (out.float() - reference).square().mean().sqrt() / reference.square().mean().sqrt()
    cosine = F.cosine_similarity(out.float().flatten(), reference.flatten(), dim=0)
    assert torch.isfinite(out).all()
    assert nrmse.item() < 0.08
    assert cosine.item() > 0.995
