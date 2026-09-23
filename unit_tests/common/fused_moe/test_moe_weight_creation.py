"""Real CUDA weight creation and checkpoint loading for singleton MoE experts."""
import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights import base_weight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe import fused_moe_weight
from lightllm.common.quantization import quantize_method
from lightllm.common.quantization.awq import AWQW4A16QuantizationMethod, HAS_VLLM
from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod, HAS_DEEPGEMM
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.common.quantization.w8a8gx import FP8w8a8g128QuantizationMethod
from lightllm.server.core.objs.start_args_type import StartArgs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture
def device_id(monkeypatch):
    device_id = torch.cuda.current_device()
    monkeypatch.setattr(quantize_method, "get_current_device_id", lambda: device_id)
    return device_id


@pytest.mark.parametrize("num_experts", [None, 1, 3], ids=["dense", "single", "multi"])
@pytest.mark.parametrize(
    "method_class,shapes,split_dims",
    [
        (NoQuantization, ((256, 128), None, None), (-2, None, None)),
        (DeepGEMMFP8w8a8B128QuantizationMethod, ((256, 128), (2, 1), None), (-2, -2, None)),
        (FP8w8a8g128QuantizationMethod, ((256, 128), (256,), None), (-2, -1, None)),
        (AWQW4A16QuantizationMethod, ((128, 32), (1, 256), (1, 32)), (-1, -1, -1)),
    ],
    ids=["bf16", "fp8-block", "fp8-channel", "awq"],
)
def test_weight_creation_preserves_expert_axis_and_loading_views(
    device_id, method_class, shapes, split_dims, num_experts
):
    if method_class is DeepGEMMFP8w8a8B128QuantizationMethod and not HAS_DEEPGEMM:
        pytest.skip("DeepGEMM is required")
    if method_class is AWQW4A16QuantizationMethod and not HAS_VLLM:
        pytest.skip("vLLM is required for AWQ")
    method = method_class()
    method.hf_quantization_config = {"group_size": 128}
    kwargs = dict(out_dims=[128, 128], in_dim=128, dtype=torch.bfloat16, device_id=device_id)
    if num_experts is None:
        parent, children = method.create_weight(**kwargs)
        prefix = ()
    else:
        parent, children = method.create_moe_weight(**kwargs, num_experts=num_experts)
        prefix = (num_experts,)
    assert len(children) == 2
    fields = ("weight", "weight_scale", "weight_zero_point")
    loaders = (method.load_weight, method.load_weight_scale, method.load_weight_zero_point)
    for name, shape, split_dim in zip(fields, shapes, split_dims):
        tensor = getattr(parent, name)
        if shape is None:
            assert tensor is None
            assert all(getattr(child, name) is None for child in children)
            continue
        assert tensor.shape == prefix + shape
        assert tensor.device == torch.device("cuda", device_id)
        child_shape = list(shape)
        child_shape[split_dim] //= 2
        for child in children:
            view = getattr(child, name)
            assert view.shape == prefix + tuple(child_shape)
            assert view.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()

    # Loading each projection/expert must update the original fused buffers.
    for expert_idx in range(1 if num_experts is None else num_experts):
        parent_expert = parent if num_experts is None else parent.get_expert(expert_idx)
        for child_idx, child in enumerate(children):
            child_expert = child if num_experts is None else child.get_expert(expert_idx)
            for name, loader in zip(fields, loaders):
                tensor = getattr(child_expert, name)
                if tensor is not None:
                    loader(torch.full_like(tensor, expert_idx * 2 + child_idx + 1), child_expert)
            assert all(child_expert.load_ok)
        for name, shape, split_dim in zip(fields, shapes, split_dims):
            if shape is None:
                continue
            child_shape = list(shape)
            child_shape[split_dim] //= 2
            expected = torch.cat(
                [
                    torch.full(child_shape, expert_idx * 2 + child_idx + 1, device=f"cuda:{device_id}")
                    for child_idx in range(2)
                ],
                dim=split_dim,
            )
            torch.testing.assert_close(getattr(parent_expert, name).float(), expected.float())


@pytest.mark.parametrize("mode,redundant", [("full", 0), ("full", 2), ("redundant", 2)])
def test_eplb_one_primary_expert_fp8_constructor_and_checkpoint_load(monkeypatch, device_id, mode, redundant):
    if not HAS_DEEPGEMM:
        pytest.skip("DeepGEMM is required")
    args = StartArgs(
        run_mode="prefill",
        tp=8,
        dp=8,
        nnodes=1,
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        eplb_placement_mode=mode,
        eplb_num_redundant_experts_per_rank=redundant,
        expert_dtype="fp8",
        data_type="bfloat16",
    )
    monkeypatch.setattr(fused_moe_weight, "get_env_start_args", lambda: args)
    monkeypatch.setattr(fused_moe_weight, "get_global_rank", lambda: 0)
    monkeypatch.setattr(fused_moe_weight, "get_global_world_size", lambda: 8)
    monkeypatch.setattr(fused_moe_weight, "get_node_world_size", lambda: 8)
    monkeypatch.setattr(base_weight, "get_dp_world_size", lambda: 1)
    monkeypatch.setattr(base_weight, "get_current_rank_in_dp", lambda: 0)
    monkeypatch.setattr(base_weight, "get_current_device_id", lambda: device_id)
    weight = fused_moe_weight.FusedMoeWeight(
        gate_proj_name="w1",
        down_proj_name="w2",
        up_proj_name="w3",
        e_score_correction_bias_name="",
        weight_prefix="model.experts",
        n_routed_experts=8,
        hidden_size=128,
        moe_intermediate_size=128,
        data_type=torch.bfloat16,
        quant_method=DeepGEMMFP8w8a8B128QuantizationMethod(),
        network_config={"num_experts_per_tok": 2},
    )
    assert weight.local_n_routed_experts == 1 + redundant
    assert weight.w13.weight.shape == (1 + redundant, 256, 128)
    assert weight.w2.weight.shape == (1 + redundant, 128, 128)
    assert weight.w13.weight_scale.shape == (1 + redundant, 2, 1)
    assert weight.w2.weight_scale.shape == (1 + redundant, 1, 1)
    checkpoint = {}
    for expert_idx in range(1 + redundant):
        for projection_idx, name in enumerate(("w1", "w2", "w3")):
            prefix = f"model.experts.{expert_idx}.{name}"
            checkpoint[prefix + ".weight"] = torch.full(
                (128, 128), expert_idx * 3 + projection_idx + 1, dtype=torch.float8_e4m3fn
            )
            checkpoint[prefix + ".weight_scale_inv"] = torch.full((1, 1), expert_idx * 3 + projection_idx + 1.0)
    weight.load_hf_weights(checkpoint)
    for projection_idx, packs in enumerate((weight.w1_list, weight.w2_list, weight.w3_list)):
        for expert_idx, pack in enumerate(packs):
            value = expert_idx * 3 + projection_idx + 1
            assert all(pack.load_ok)
            torch.testing.assert_close(
                pack.weight.float(), torch.full((128, 128), value, device=f"cuda:{device_id}").float()
            )
            torch.testing.assert_close(pack.weight_scale, torch.full((1, 1), float(value), device=f"cuda:{device_id}"))
