from types import SimpleNamespace

import pytest
import torch

from lightllm.common.quantization import Quantcfg, deepgemm
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.utils.envs_utils import get_env_use_ue8m0_activation_scales, get_env_use_ue8m0_scales


@pytest.mark.parametrize("scale_format", [None, "auto", "fp32", "ue8m0", " FP32 "])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("default", [None, False, True])
def test_activation_override_preserves_weight_policy(monkeypatch, scale_format, legacy, default):
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", str(int(legacy)))
    if scale_format is None:
        monkeypatch.delenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", raising=False)
    else:
        monkeypatch.setenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", scale_format)
    expected = legacy if default is None else default
    if scale_format is not None and scale_format.strip().lower() != "auto":
        expected = scale_format.strip().lower() == "ue8m0"
    assert get_env_use_ue8m0_activation_scales(default) == expected
    assert get_env_use_ue8m0_scales() == legacy


def test_invalid_or_unsupported_activation_format_fails_explicitly(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", "typo")
    with pytest.raises(ValueError, match="LIGHTLLM_ACTIVATION_SCALE_FORMAT"):
        get_env_use_ue8m0_activation_scales()
    monkeypatch.setenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", "fp32")
    with pytest.raises(ValueError, match="requires UE8M0"):
        get_env_use_ue8m0_activation_scales(True, require_ue8m0=True)


@pytest.mark.skipif(not torch.cuda.is_available() or not deepgemm.HAS_DEEPGEMM, reason="CUDA + DeepGEMM required")
@pytest.mark.parametrize("weight_ue8m0", [False, True])
def test_dense_activation_override_reuses_exact_weight_pack(monkeypatch, weight_ue8m0):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Continuous FP32 activation scales target Hopper")
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "0")
    config = {
        "n_layer": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0" if weight_ue8m0 else "float32",
        },
    }
    method = Quantcfg(config).get_quant_method(0, "q_proj")
    pack, _ = method.create_weight([128], 512, torch.bfloat16, 0)
    torch.manual_seed(54)
    method.load_weight(torch.randn(128, 512, device="cuda", dtype=torch.bfloat16), pack)
    before_weight, before_scale = pack.weight.view(torch.uint8).clone(), pack.weight_scale.clone()
    x = torch.randn(17, 512, device="cuda", dtype=torch.bfloat16)
    quantize = deepgemm.per_token_group_quant_fp8
    calls = []

    def capture(values, group_size, **kwargs):
        result = quantize(values, group_size, **kwargs)
        expected = values.float().reshape(17, 4, 128).abs().amax(-1).clamp_min(1e-10) / 448
        if kwargs["use_ue8m0_scales"]:
            expected = expected.log2().ceil().exp2()
        assert group_size == 128 and result[1].dtype == torch.float32
        torch.testing.assert_close(result[1], expected, rtol=1e-6, atol=0)
        calls.append(kwargs["use_ue8m0_scales"])
        return result

    monkeypatch.setattr(deepgemm, "per_token_group_quant_fp8", capture)
    for scale_format in ("auto", "fp32", "ue8m0"):
        monkeypatch.setenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", scale_format)
        result = method.apply(x, pack, use_custom_tensor_mananger=False)
        assert torch.isfinite(result).all()
        assert torch.equal(pack.weight.view(torch.uint8), before_weight)
        assert torch.equal(pack.weight_scale, before_scale)
        assert method.use_ue8m0_scales == weight_ue8m0
    assert calls == [weight_ue8m0, False, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scale_format", ["auto", "fp32", "ue8m0"])
def test_fallback_block_quantization_honors_override(monkeypatch, scale_format):
    from lightllm.common.quantization.w8a8 import FP8w8a8B128QuantizationMethod

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    monkeypatch.setenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", scale_format)
    method = object.__new__(FP8w8a8B128QuantizationMethod)
    method.block_size = 128
    x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    pack = WeightPack(
        weight=torch.ones(128, 512, device="cuda", dtype=torch.float8_e4m3fn),
        weight_scale=torch.full((1, 4), 0.125, device="cuda"),
    )
    result = method._dynamic_quant_input(x, pack, use_custom_tensor_mananger=False)
    expected = x.float().reshape(3, 4, 128).abs().amax(-1).clamp_min(1e-10) / 448
    if scale_format == "ue8m0":
        expected = expected.log2().ceil().exp2()
    torch.testing.assert_close(result[2], expected, rtol=1e-6, atol=0)
    assert torch.all(pack.weight_scale == 0.125)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scale_format", ["auto", "fp32", "ue8m0"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_deepep_dispatch_uses_activation_override(monkeypatch, scale_format, legacy, overlap):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", str(int(legacy)))
    monkeypatch.setenv("LIGHTLLM_ACTIVATION_SCALE_FORMAT", scale_format)
    expected = legacy if scale_format == "auto" else scale_format == "ue8m0"
    hidden = torch.zeros(3, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(3, 1, device="cuda", dtype=torch.int64)
    weights = torch.ones(3, 1, device="cuda")
    calls = []

    def dispatch(*args, **kwargs):
        calls.append(kwargs["round_scale"])
        return object(), object(), object(), None, None

    buffer = SimpleNamespace(
        group_size=1,
        low_latency_dispatch=dispatch,
        low_latency_combine=lambda *args, **kwargs: (hidden, None, None),
    )
    monkeypatch.setattr(ep.dist_group_manager, "ep_low_latency_buffer", buffer, raising=False)
    if overlap:
        impl = object.__new__(FuseMoeDeepGEMM)
        impl.quant_method = SimpleNamespace(method_name="fp8w8a8-b128-deepgemm")
        impl.total_expert_num_contain_redundancy = 1
        impl._select_experts = lambda **kwargs: (weights, ids, ids)
        impl.low_latency_dispatch(hidden, hidden, None, False, 1, True, 1, 1, "sigmoid")
    else:
        monkeypatch.setattr(ep, "masked_group_gemm", lambda *args, **kwargs: hidden)
        ep.fused_experts_impl(
            hidden,
            torch.zeros(1, 256, 128, device="cuda", dtype=torch.float8_e4m3fn),
            torch.zeros(1, 128, 128, device="cuda", dtype=torch.float8_e4m3fn),
            weights,
            ids,
            buffer=buffer,
            is_prefill=False,
            num_experts=1,
            use_fp8_w8a8=True,
            use_fp8_all2all=True,
            block_size_k=128,
        )
    assert calls == [expected]
