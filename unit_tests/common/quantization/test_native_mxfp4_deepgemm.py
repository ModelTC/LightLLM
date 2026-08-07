import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.deepgemm import (
    DeepGEMMNativeMXFP4B32QuantizationMethod,
    pack_ue8m0_scales_to_int32,
)
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.distributed import communication_op
from lightllm.distributed.communication_op import DistributeGroupManager
from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight


def test_pack_ue8m0_scales_matches_deepgemm_byte_order():
    scales = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.uint8)

    packed = pack_ue8m0_scales_to_int32(scales)
    import deep_gemm

    deepgemm_input = (scales.to(torch.int32) << 23).view(torch.float32)

    assert packed.dtype == torch.int32
    assert packed.shape == (1, 2)
    assert torch.equal(packed.view(torch.uint8), scales)
    assert torch.equal(packed, deep_gemm.pack_ue8m0_to_int(deepgemm_input))
    assert torch.equal(packed, pack_ue8m0_scales_to_int32(scales.view(torch.float8_e8m0fnu)))


@pytest.mark.parametrize("shape", [(8,), (2, 6)])
def test_pack_ue8m0_scales_validates_shape(shape):
    with pytest.raises(ValueError):
        pack_ue8m0_scales_to_int32(torch.zeros(shape, dtype=torch.uint8))


def test_native_mxfp4_load_preserves_weight_bytes_and_packs_scales():
    quant_method = DeepGEMMNativeMXFP4B32QuantizationMethod.__new__(DeepGEMMNativeMXFP4B32QuantizationMethod)
    weight_pack = WeightPack(
        weight=torch.empty((2, 256), dtype=torch.int8),
        weight_scale=torch.empty_strided((2, 4), (1, 2), dtype=torch.int32),
    )
    weight = torch.arange(512, dtype=torch.int64).to(torch.uint8).reshape(2, 256)
    scale = torch.arange(32, dtype=torch.uint8).reshape(2, 16)

    quant_method.load_weight(weight, weight_pack)
    quant_method.load_weight_scale(scale, weight_pack)

    assert torch.equal(weight_pack.weight.view(torch.uint8), weight)
    assert torch.equal(weight_pack.weight_scale, pack_ue8m0_scales_to_int32(scale))
    assert weight_pack.load_ok[:2] == [True, True]


def test_native_mxfp4_load_validates_weight_and_scale():
    quant_method = DeepGEMMNativeMXFP4B32QuantizationMethod.__new__(DeepGEMMNativeMXFP4B32QuantizationMethod)
    weight_pack = WeightPack(
        weight=torch.empty((2, 64), dtype=torch.int8),
        weight_scale=torch.empty((2, 4), dtype=torch.int32),
    )

    with pytest.raises(TypeError, match="packed weights"):
        quant_method.load_weight(torch.zeros((2, 64), dtype=torch.float16), weight_pack)
    with pytest.raises(ValueError, match="weight shape mismatch"):
        quant_method.load_weight(torch.zeros((2, 63), dtype=torch.uint8), weight_pack)
    with pytest.raises(TypeError, match="float8_e8m0fnu"):
        quant_method.load_weight_scale(torch.zeros((2, 16), dtype=torch.float32), weight_pack)
    with pytest.raises(ValueError, match="scale shape mismatch"):
        quant_method.load_weight_scale(torch.zeros((2, 7), dtype=torch.uint8), weight_pack)


@pytest.mark.parametrize(
    ("config_expert_dtype", "start_arg_expert_dtype"),
    [("fp4", None), ("mxfp4", None), (None, "fp4"), (None, "mxfp4")],
)
def test_quantcfg_routes_deepseek_v4_native_mxfp4(monkeypatch, config_expert_dtype, start_arg_expert_dtype):
    monkeypatch.setattr("lightllm.common.quantization.is_sm100_gpu", lambda: True)
    config = {"n_layer": 2, "model_type": "deepseek_v4"}
    if config_expert_dtype is not None:
        config["expert_dtype"] = config_expert_dtype

    ep_quant_cfg = Quantcfg(config, expert_dtype=start_arg_expert_dtype, enable_ep_moe=True)
    tp_quant_cfg = Quantcfg(config, expert_dtype=start_arg_expert_dtype, enable_ep_moe=False)

    assert ep_quant_cfg.get_quant_type(0, "fused_moe") == "mxfp4fp8-b32-deepgemm"
    assert tp_quant_cfg.get_quant_type(0, "fused_moe") == "mxfp4w4a16-b32-marlin"


def test_quantcfg_rejects_deepseek_v4_native_mxfp4_ep_without_sm100(monkeypatch):
    monkeypatch.setattr("lightllm.common.quantization.is_sm100_gpu", lambda: False)

    with pytest.raises(RuntimeError, match="native MXFP4 experts with --enable_ep_moe require an SM100 GPU"):
        Quantcfg({"n_layer": 1, "model_type": "deepseek_v4", "expert_dtype": "mxfp4"}, enable_ep_moe=True)


def test_quantcfg_keeps_other_model_fp4_route(monkeypatch):
    monkeypatch.setattr("lightllm.common.quantization.is_sm100_gpu", lambda: True)

    quant_cfg = Quantcfg({"n_layer": 1, "model_type": "qwen3_moe", "expert_dtype": "fp4"}, enable_ep_moe=True)

    assert quant_cfg.get_quant_type(0, "fused_moe") == "fp4fp8-b32-deepgemm"


def test_quantcfg_keeps_fp8_deepseek_v4_checkpoint_on_online_fp4_route(monkeypatch):
    monkeypatch.setattr("lightllm.common.quantization.is_sm100_gpu", lambda: True)
    config = {
        "n_layer": 1,
        "model_type": "deepseek_v4",
        "expert_dtype": "fp8",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }

    quant_cfg = Quantcfg(config, expert_dtype="fp4", enable_ep_moe=True)
    fp8_quant_cfg = Quantcfg(config, enable_ep_moe=True)

    assert quant_cfg.get_quant_type(0, "fused_moe") == "fp4fp8-b32-deepgemm"
    assert fp8_quant_cfg.get_quant_type(0, "fused_moe") == "fp8w8a8-b128-deepgemm"


def _make_fp8_expert_pairing_layer():
    layer = DeepseekV4TransformerLayerWeight.__new__(DeepseekV4TransformerLayerWeight)
    layer.prefix = "layers.0"
    layer.lock = threading.Lock()
    layer.data_type_ = torch.bfloat16
    layer.quant_cfg = SimpleNamespace(config_expert_dtype="fp8")
    layer.experts_ = SimpleNamespace(
        quant_method=SimpleNamespace(weight_scale_suffix=None),
        local_expert_ids=[1],
    )
    return layer


@pytest.mark.parametrize("first_kind", ["weight", "scale"])
def test_dequant_local_fp8_experts_pairs_cross_shard_in_any_order(monkeypatch, first_kind):
    layer = _make_fp8_expert_pairing_layer()
    calls = []

    def fake_dequant(weight, scale):
        calls.append((weight, scale))
        return weight.to(torch.bfloat16) + scale.to(torch.bfloat16)

    monkeypatch.setattr(
        "lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight.dequant_fp8_block_to_bf16",
        fake_dequant,
    )
    second_kind = "scale" if first_kind == "weight" else "weight"
    local_prefix = "layers.0.ffn.experts.1.w1"
    remote_prefix = "layers.0.ffn.experts.2.w1"
    first = {
        f"{local_prefix}.{first_kind}": torch.tensor([2.0]),
        f"{remote_prefix}.{first_kind}": torch.tensor([9.0]),
    }
    second = {
        f"{local_prefix}.{second_kind}": torch.tensor([3.0]),
        f"{remote_prefix}.{second_kind}": torch.tensor([8.0]),
    }

    layer._dequant_local_fp8_experts_to_bf16(first)
    layer._dequant_local_fp8_experts_to_bf16(second)

    assert len(calls) == 1
    assert f"{local_prefix}.weight" in second
    assert second[f"{local_prefix}.weight"].dtype == torch.bfloat16
    assert f"{remote_prefix}.{second_kind}" in second
    assert layer._pending_fp8_expert_pairs == {}


def test_get_moe_quant_methods_supports_deepseek_v4_experts_attribute():
    quant_method = SimpleNamespace(method_name="fp4fp8-b32-deepgemm")
    layer_weight = SimpleNamespace(experts_=SimpleNamespace(quant_method=quant_method))

    assert DistributeGroupManager.get_moe_quant_methods([layer_weight]) == {"fp4fp8-b32-deepgemm"}


@pytest.mark.parametrize(
    ("expert_quant_method_names", "expected_elastic", "expected_legacy", "expected_mega"),
    [
        ({"fp4fp8-b32-deepgemm"}, 0, 0, 1),
        ({"fp8w8a8-b128-deepgemm"}, 1, 1, 0),
        ({"fp4fp8-b32-deepgemm", "fp8w8a8-b128-deepgemm"}, 1, 1, 1),
    ],
)
def test_deepep_buffers_only_construct_elastic_for_legacy_path(
    monkeypatch,
    expert_quant_method_names,
    expected_elastic,
    expected_legacy,
    expected_mega,
):
    calls = {"elastic": [], "legacy": [], "mega": [], "sms": []}

    class FakeElasticBuffer:
        def __init__(self, *args, **kwargs):
            calls["elastic"].append((args, kwargs))

        def get_theoretical_num_sms(self, *args):
            return 42

    class FakeBuffer:
        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            return 123

        def __init__(self, *args, **kwargs):
            calls["legacy"].append((args, kwargs))

    fake_deep_ep = SimpleNamespace(ElasticBuffer=FakeElasticBuffer, Buffer=FakeBuffer)
    fake_deep_gemm = SimpleNamespace(
        get_symm_buffer_for_mega_moe=lambda *args: calls["mega"].append(args) or "mega_buffer"
    )
    monkeypatch.setattr(communication_op, "HAS_DEEPEP", True)
    monkeypatch.setattr(communication_op, "deep_ep", fake_deep_ep)
    monkeypatch.setattr(communication_op, "get_env_start_args", lambda: SimpleNamespace(enable_ep_moe=True))
    monkeypatch.setattr(communication_op, "get_deepep_num_max_dispatch_tokens_per_rank_prefill", lambda: 64)
    monkeypatch.setattr(communication_op, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 32)
    monkeypatch.setattr(communication_op, "get_global_world_size", lambda: 1)
    monkeypatch.setattr(communication_op, "get_redundancy_expert_num", lambda: 0)
    monkeypatch.setattr(communication_op, "is_sm100_gpu", lambda: True)
    monkeypatch.setattr(communication_op.dist, "new_group", lambda ranks: "group")
    monkeypatch.setattr(communication_op.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(communication_op.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setitem(sys.modules, "deep_gemm", fake_deep_gemm)

    manager = DistributeGroupManager()
    manager._set_num_sms_for_deep_gemm = calls["sms"].append
    manager.new_deepep_group(
        n_routed_experts=8,
        hidden_size=16,
        expert_quant_method_names=expert_quant_method_names,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )

    assert len(calls["elastic"]) == expected_elastic
    assert len(calls["legacy"]) == expected_legacy
    assert len(calls["mega"]) == expected_mega
    assert calls["sms"] == ([42] if expected_elastic else [0])
