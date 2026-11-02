import json

import pytest
import torch
import yaml

from lightllm.common.quantization import Quantcfg


def selection(implementation="triton", weight_format="fp8_block128", activation_format="fp8_group128"):
    return dict(weight_format=weight_format, activation_format=activation_format, implementation=implementation)


def test_checkpoint_encoding_is_independent_of_explicit_implementation(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "0")
    checkpoint = dict(quant_method="fp8", weight_block_size=[128, 128], scale_fmt="ue8m0")
    path = tmp_path / "quant.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                quant_type=selection(),
                mix_bits=[dict(name="down_proj", layer_nums=[1], quant_type=selection("deepgemm"))],
            )
        )
    )
    config = Quantcfg(dict(n_layer=2, quantization_config=checkpoint), custom_cfg_path=path)
    first = config.get_weight_plan(0, "down_proj")
    second = config.get_weight_plan(1, "down_proj")
    assert first.method_key == "fp8w8a8-b128-triton"
    assert second.method_key == "fp8w8a8-b128-deepgemm"
    assert first.checkpoint == second.checkpoint
    assert first.selection.weight_format == second.selection.weight_format == "fp8_block128"
    checkpoint["weight_block_size"][0] = 64
    assert json.loads(first.checkpoint.parameters)["weight_block_size"] == [128, 128]


def test_unknown_representation_pair_and_checkpoint_conversion_fail_before_loading():
    with pytest.raises(ValueError, match="no quantization provider"):
        Quantcfg(dict(n_layer=1), quant_type=selection("marlin")).get_weight_plan(0, "q")
    with pytest.raises(ValueError, match="AWQ checkpoint"):
        Quantcfg(dict(n_layer=1), quant_type=selection("vllm", "awq_int4_gemm", "plain")).get_weight_plan(0, "q")
    with pytest.raises(ValueError, match="matching weight representation"):
        Quantcfg(
            dict(n_layer=1, quantization_config=dict(quant_method="fp8", weight_block_size=[128, 128])),
            quant_type=selection("triton", "fp8_per_tensor", "fp8_per_token"),
        ).get_weight_plan(0, "q")


def test_scale_rounding_is_frozen_per_model_before_binding(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "0")
    main = Quantcfg(
        dict(n_layer=1, quantization_config=dict(quant_method="fp8", weight_block_size=[128, 128], scale_fmt="ue8m0")),
        quant_type=selection("deepgemm"),
    )
    draft = Quantcfg(dict(n_layer=1), quant_type=selection("deepgemm"))
    before = draft.content_identity()
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", "1")
    assert main.get_weight_plan(0, "q").use_ue8m0_scales
    assert not draft.get_weight_plan(0, "q").use_ue8m0_scales
    assert draft.content_identity() == before


@pytest.mark.parametrize("round_scale", [False, True])
def test_low_latency_dispatch_uses_bound_scale_rounding(monkeypatch, round_scale):
    from types import SimpleNamespace
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
    from lightllm.distributed import dist_group_manager

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", str(int(not round_scale)))
    seen = []

    def dispatch(**kwargs):
        seen.append(kwargs["round_scale"])
        return (None,) * 5

    monkeypatch.setattr(
        dist_group_manager, "ep_low_latency_buffer", SimpleNamespace(low_latency_dispatch=dispatch), raising=False
    )
    impl = FuseMoeDeepGEMM.__new__(FuseMoeDeepGEMM)
    impl.quant_method = SimpleNamespace(method_name="fp8w8a8-b128-deepgemm", use_ue8m0_scales=round_scale)
    impl.total_expert_num_contain_redundancy = 2
    impl.low_latency_dispatch_with_topk(torch.ones(1, 128), torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1))
    assert seen == [round_scale]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_structured_selection_loads_and_executes_existing_fp8_kernel(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    torch.manual_seed(35)
    input = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    outputs, packs = [], []
    for value in ("fp8w8a8-b128-triton", selection()):
        method = Quantcfg(dict(n_layer=1), quant_type=value).get_quant_method(0, "q")
        pack, _ = method.create_weight([256], 128, torch.bfloat16, 0)
        method.load_weight(weight, pack)
        assert all(pack.load_ok)
        assert method.weight_plan.checkpoint.encoding == "plain"
        packs.append(pack)
        outputs.append(method.apply(input, pack, use_custom_tensor_mananger=False))
    assert torch.equal(packs[0].weight.view(torch.uint8), packs[1].weight.view(torch.uint8))
    assert torch.equal(packs[0].weight_scale, packs[1].weight_scale)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
