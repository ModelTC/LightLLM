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


@pytest.fixture
def packed_weight_layout(monkeypatch):
    calls = []
    transform = deepgemm.deep_gemm.transform_sf_into_required_layout
    sm100 = deepgemm.is_sm100_gpu()

    def record_transform(sf, n, k, recipe, num_groups=None, is_sfa=False):
        calls.append((n, k, num_groups))
        if sm100:
            return transform(sf, n, k, recipe, num_groups=num_groups, is_sfa=is_sfa)
        # Exercise DeepGEMM's actual pack kernel on Hopper, emulating only the SM100 layout dispatch.
        assert recipe == (1, 128, 128) and not is_sfa
        expanded = sf.index_select(-2, torch.arange(n, device=sf.device) // 128)
        return deepgemm.deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(expanded)

    monkeypatch.setattr(deepgemm.deep_gemm, "transform_sf_into_required_layout", record_transform)
    return calls


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
def test_deepgemm_selects_scale_format_for_machine(monkeypatch, sm100, packed_weight_layout):
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    monkeypatch.setattr(deepgemm, "is_sm100_gpu", lambda: sm100)
    method = Quantcfg({"n_layer": 1}, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    weight_pack, _ = method.create_weight([128], 512, torch.bfloat16, 0)
    method.load_weight(torch.ones((128, 512), dtype=torch.bfloat16, device="cuda"), weight_pack)
    seen = []

    def check_gemm_inputs(a, b, out):
        seen.append(a[1].dtype)
        assert b[1].dtype == (torch.int32 if sm100 else torch.float32)
        scales = _unpack_scales(a[1], 4) if sm100 else a[1]
        torch.testing.assert_close(scales, torch.full_like(scales, 2.0 ** -8), rtol=0, atol=0)
        weight_scales = _unpack_scales(b[1], 4) if sm100 else b[1]
        torch.testing.assert_close(weight_scales, torch.full_like(weight_scales, 2.0 ** -8), rtol=0, atol=0)
        out.zero_()

    # Exercise format selection and real quantization on Hopper; SM100 GEMM needs a separate machine.
    monkeypatch.setattr(deepgemm, "_deepgemm_fp8_nt", check_gemm_inputs)
    method.apply(
        torch.ones((3, 512), dtype=torch.bfloat16, device="cuda"), weight_pack, use_custom_tensor_mananger=False
    )
    method.apply(
        torch.ones((3, 512), dtype=torch.bfloat16, device="cuda"), weight_pack, use_custom_tensor_mananger=False
    )
    assert seen == [torch.int32 if sm100 else torch.float32] * 2
    assert len(packed_weight_layout) == int(sm100)


@pytest.mark.skipif(not deepgemm.HAS_DEEPGEMM, reason="requires DeepGEMM")
@pytest.mark.parametrize("sm100", [False, True])
@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("experts", [None, 2])
def test_weight_scale_loading_preserves_fused_views(
    monkeypatch, packed_weight_layout, sm100, ue8m0, prequantized, experts
):
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", str(int(ue8m0)))
    monkeypatch.setattr(deepgemm, "is_sm100_gpu", lambda: sm100)
    method = Quantcfg({"n_layer": 1}, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    args = dict(out_dims=[128, 129], in_dim=640, dtype=torch.bfloat16, device_id=0)
    full, parts = (
        method.create_weight(**args) if experts is None else method.create_moe_weight(**args, num_experts=experts)
    )
    packed = sm100 and ue8m0
    expected_parts = []
    torch.manual_seed(20261008)
    for part in parts:
        raw = torch.randn_like(part.weight, dtype=torch.bfloat16)
        raw[..., 128:256] *= 8
        q, scales = weight.weight_quant(raw, use_ue8m0_scales=ue8m0)
        loads = (
            [(part, raw, q, scales)]
            if experts is None
            else [(part.get_expert(i), raw[i], q[i], scales[i]) for i in range(experts)]
        )
        for dest, bf16, fp8, sf in loads:
            if prequantized:
                method.load_weight(fp8, dest)
                method.load_weight_scale(sf.cpu(), dest)
            else:
                method.load_weight(bf16, dest)
            assert all(dest.load_ok)
        torch.testing.assert_close(part.weight.float(), q.float(), rtol=0, atol=0)
        expected = (
            scales.index_select(-2, torch.arange(part.weight.shape[-2], device="cuda") // 128) if packed else scales
        )
        flat = part.weight_scale.reshape(-1, part.weight_scale.shape[-1])
        actual = _unpack_scales(flat, 5).reshape_as(expected) if packed else part.weight_scale
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_parts.append(expected)

    expected = torch.cat(expected_parts, dim=-2)
    assert full.weight_scale.dtype == (torch.int32 if packed else torch.float32)
    if packed:
        assert full.weight_scale.shape[-2:] == (257, 2)
        assert full.weight_scale.stride()[-2:] == (1, 260)
        actual = _unpack_scales(full.weight_scale.reshape(-1, 2), 5).reshape_as(expected)
    else:
        actual = full.weight_scale
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(packed_weight_layout) == (2 * (experts or 1) if packed else 0)


@pytest.mark.skipif(not deepgemm.HAS_DEEPGEMM, reason="requires DeepGEMM")
@pytest.mark.parametrize("need_trans", [False, True])
@pytest.mark.parametrize("k", [512, 640])
def test_tp_moe_consumes_packed_weight_scales(monkeypatch, packed_weight_layout, need_trans, k):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import grouped_matmul

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    monkeypatch.setattr(deepgemm, "is_sm100_gpu", lambda: True)
    method = Quantcfg({"n_layer": 1}, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    full, _ = method.create_moe_weight([256], k, torch.bfloat16, 0, 2)
    torch.manual_seed(20261008)
    raw = torch.randn((2, 256, k), device="cuda", dtype=torch.bfloat16)
    q, scales = weight.weight_quant(raw, use_ue8m0_scales=True)
    method.load_weight(raw, full)
    x = torch.randn((17, k), device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([9, 8], device="cuda", dtype=torch.int32)
    indices = torch.zeros((2, 17), device="cuda", dtype=torch.int32)
    indices[0, :9] = torch.arange(9, device="cuda")
    indices[1, :8] = torch.arange(9, 17, device="cuda")
    outputs = []
    for sf in (scales, full.weight_scale):
        out = torch.empty((17, 256), device="cuda", dtype=torch.bfloat16)
        grouped_matmul(
            17,
            x,
            None,
            counts,
            indices,
            torch.ones((2, 17), device="cuda"),
            q,
            sf,
            1,
            out,
            mul_routed_weight=True,
            use_fp8_w8a8=True,
            run_config=dict(
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
                GROUP_SIZE_M=1,
                num_warps=4,
                num_stages=2,
                NEED_TRANS=need_trans,
            ),
        )
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


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
    weight_scales = weight_pack.weight_scale
    if method.use_packed_ue8m0:
        weight_scales = _unpack_scales(weight_scales, 8)[::128]
    assert torch.all(weight_scales > 0)
    log_scales = torch.log2(weight_scales)
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


@pytest.mark.skipif(not deepgemm.HAS_DEEPGEMM, reason="requires DeepGEMM")
@pytest.mark.parametrize("prepacked", [False, True])
@pytest.mark.parametrize("sm100", [False, True])
def test_redundancy_expert_update_packs_scales(monkeypatch, packed_weight_layout, prepacked, sm100):
    from types import SimpleNamespace
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.ep_redundancy import (
        FusedMoeWeightEPAutoRedundancy,
    )

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    monkeypatch.setattr(deepgemm, "is_sm100_gpu", lambda: sm100)
    method = Quantcfg({"n_layer": 1}, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    w13, _ = method.create_moe_weight([128, 128], 512, torch.bfloat16, 0, 4)
    w2, _ = method.create_moe_weight([512], 128, torch.bfloat16, 0, 4)
    update = object.__new__(FusedMoeWeightEPAutoRedundancy)
    update.redundancy_expert_num = 2
    update.redundancy_expert_ids = [6, 7]
    update._ep_w = SimpleNamespace(
        w13=w13,
        w2=w2,
        quant_method=method,
        redundancy_expert_ids_tensor=torch.zeros(2, device="cuda", dtype=torch.int64),
    )
    expected = []
    for dest in (w13, w2):
        dest.weight.zero_()
        dest.weight_scale.zero_()
        shape = (2,) + dest.weight.shape[-2:]
        raw = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        q, sf = weight.weight_quant(raw, use_ue8m0_scales=True)
        if prepacked:
            source, _ = method.create_moe_weight([shape[-2]], shape[-1], torch.bfloat16, 0, 2)
            method.load_weight(raw, source)
            sf = source.weight_scale
        expected.append([q, sf])
    update.w13, update.w2 = expected
    before = len(packed_weight_layout)
    update.commit()
    assert len(packed_weight_layout) == before + (2 if sm100 and not prepacked else 0)
    for dest, (q, sf) in zip((w13, w2), expected):
        assert torch.count_nonzero(dest.weight[:2].float()) == 0
        assert torch.count_nonzero(dest.weight_scale[:2]) == 0
        torch.testing.assert_close(dest.weight[2:].float(), q.float(), rtol=0, atol=0)
        if sm100:
            groups = (q.shape[-1] + 127) // 128
            actual = _unpack_scales(dest.weight_scale[2:].reshape(-1, dest.weight_scale.shape[-1]), groups)
            expected_sf = (
                _unpack_scales(sf.reshape(-1, sf.shape[-1]), groups)
                if prepacked
                else sf.repeat_interleave(128, dim=-2).reshape(-1, groups)
            )
        else:
            actual, expected_sf = dest.weight_scale[2:], sf
        torch.testing.assert_close(actual, expected_sf, rtol=0, atol=0)
    assert update._ep_w.redundancy_expert_ids_tensor.tolist() == [6, 7]


@pytest.mark.skipif(not deepgemm.HAS_DEEPGEMM, reason="requires DeepGEMM")
@pytest.mark.parametrize("is_prefill", [False, True])
def test_ep_moe_uses_logical_block_size_with_packed_weights(monkeypatch, is_prefill):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    monkeypatch.setattr(deepgemm, "is_sm100_gpu", lambda: True)
    method = Quantcfg({"n_layer": 1}, quant_type="fp8w8a8-b128-deepgemm").get_quant_method(0, "q_proj")
    w13, _ = method.create_moe_weight([128, 128], 512, torch.bfloat16, 0, 2)
    w2, _ = method.create_moe_weight([512], 128, torch.bfloat16, 0, 2)
    x = torch.randn((3, 512), device="cuda", dtype=torch.bfloat16)
    q, sf = ep.quantize_fused_experts_input(x, w13, method)
    assert q.shape == x.shape and sf.shape == (3, 4)
    seen = []
    monkeypatch.setattr(ep.dist_group_manager, "ep_buffer", object(), raising=False)
    monkeypatch.setattr(ep.dist_group_manager, "ep_low_latency_buffer", object(), raising=False)

    def check_dispatch(**kwargs):
        seen.append(kwargs["block_size_k"])
        assert kwargs["w1_scale"].dtype == kwargs["w2_scale"].dtype == torch.int32
        return x

    monkeypatch.setattr(ep, "fused_experts_impl", check_dispatch)
    ep.fused_experts(
        x,
        w13,
        w2,
        torch.ones((3, 1), device="cuda"),
        torch.zeros((3, 1), device="cuda", dtype=torch.int64),
        num_experts=2,
        quant_method=method,
        is_prefill=is_prefill,
    )
    assert seen == [128]
