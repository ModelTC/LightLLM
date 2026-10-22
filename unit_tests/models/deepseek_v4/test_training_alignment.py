from types import SimpleNamespace

import pytest
import torch
from lightllm.models.deepseek_v4.layer_weights import transformer_layer_weight


@pytest.mark.parametrize("major", [9, 10])
def test_wo_a_keeps_bf16_on_hopper_and_blackwell(monkeypatch, major):
    layer = transformer_layer_weight.DeepseekV4TransformerLayerWeight.__new__(
        transformer_layer_weight.DeepseekV4TransformerLayerWeight
    )
    layer.prefix = "layers.0"
    layer.hidden, layer.q_lora_rank, layer.o_lora_rank = 4096, 1024, 1024
    layer.n_heads, layer.head_dim, layer.o_groups, layer.tp_world_size_ = 64, 512, 8, 8
    layer.data_type_ = torch.bfloat16
    layer.get_quant_method = lambda name: SimpleNamespace(method_name="fp8w8a8-b128-deepgemm")
    for name in ("ROWMMWeight", "COLMMWeight", "ROWBMMWeight", "RMSNormWeight", "TpAttSinkWeight"):
        monkeypatch.setattr(transformer_layer_weight, name, lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (major, 0))
    layer._init_qkvo()
    assert layer.o_proj_use_mm == (major >= 10)
    assert layer.wo_a_.quant_method is None
    assert layer.wo_b_.quant_method.method_name == "fp8w8a8-b128-deepgemm"
    if not layer.o_proj_use_mm:
        assert layer.wo_a_.dim0 == 8


@pytest.mark.parametrize("use_mm", [False, True])
@pytest.mark.parametrize("stored_fp8", [False, True])
def test_wo_a_loading_preserves_bf16_weights_and_dequantizes_fp8(monkeypatch, use_mm, stored_fp8):
    layer = transformer_layer_weight.DeepseekV4TransformerLayerWeight.__new__(
        transformer_layer_weight.DeepseekV4TransformerLayerWeight
    )
    layer.prefix, layer.data_type_, layer.o_proj_use_mm = "layers.0", torch.bfloat16, use_mm
    layer.n_heads, layer.head_dim, layer.o_groups, layer.o_lora_rank = 2, 4, 2, 2
    layer.experts_ = SimpleNamespace(quant_method=SimpleNamespace(weight_scale_suffix="weight_scale_inv"))
    name = "layers.0.attn.wo_a.weight"
    source = torch.arange(16, dtype=torch.bfloat16).view(4, 4)
    weights = {name: source.to(torch.float8_e4m3fn) if stored_fp8 else source.clone()}
    if stored_fp8:
        weights["layers.0.attn.wo_a.scale"] = torch.ones(1)
    dequant_calls = []

    def dequant(weight, scale):
        dequant_calls.append(weight.dtype)
        return weight.bfloat16() * scale

    monkeypatch.setattr(transformer_layer_weight, "dequant_fp8_block_to_bf16", dequant)
    layer._dequant_in_place(weights)
    assert dequant_calls == ([torch.float8_e4m3fn] if stored_fp8 else [])
    assert list(weights) == [name]
    expected = source if use_mm else source.view(2, 2, 4).transpose(1, 2)
    torch.testing.assert_close(weights[name], expected, rtol=0, atol=0)
