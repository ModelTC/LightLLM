import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import (
    gated_rmsnorm_forward,
)
from lightllm.models.glm5_next.layer_weights.transformer_layer_weight import (
    Glm5NextMergedKdaProjection,
)
from lightllm.models.deepseek3_2.triton_kernel.act_quant import act_quant
from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import (
    hadamard_transform,
    hadamard_transform_quant_fp8,
)
from lightllm.models.deepseek3_2.triton_kernel.indexer_weight_scale import (
    scale_indexer_weights_,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_gated_rmsnorm_activations(activation):
    torch.manual_seed(0)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)

    actual = gated_rmsnorm_forward(
        x=x,
        weight=weight,
        bias=None,
        eps=1e-6,
        z=gate,
        activation=activation,
        run_config={"BLOCK_N": 128, "num_warps": 4},
    )

    x_float = x.float()
    expected = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-6)
    expected *= weight.float()
    gate_float = gate.view_as(x).float()
    if activation == "silu":
        expected *= gate_float * gate_float.sigmoid()
    else:
        expected *= gate_float.sigmoid()

    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)


def test_merged_kda_projection_mixes_sharded_and_replicated_weights(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    projection, head_count, head_dim, in_dim = 8, 4, 2, 3
    names = [f"weight_{index}" for index in range(6)]
    merged = Glm5NextMergedKdaProjection(
        in_dim=in_dim,
        projection=projection,
        head_count=head_count,
        head_dim=head_dim,
        weight_names=names,
        data_type=torch.float32,
        tp_rank=1,
        tp_world_size=2,
    )
    full_weights = {
        names[0]: torch.arange(0, 24, dtype=torch.float32).view(8, 3),
        names[1]: torch.arange(24, 48, dtype=torch.float32).view(8, 3),
        names[2]: torch.arange(48, 72, dtype=torch.float32).view(8, 3),
        names[3]: torch.arange(72, 84, dtype=torch.float32).view(4, 3),
        names[4]: torch.arange(84, 90, dtype=torch.float32).view(2, 3),
        names[5]: torch.arange(90, 96, dtype=torch.float32).view(2, 3),
    }
    merged.load_hf_weights(full_weights)

    expected_weight = torch.cat(
        [
            full_weights[names[0]][4:],
            full_weights[names[1]][4:],
            full_weights[names[2]][4:],
            full_weights[names[3]][2:],
            full_weights[names[4]],
            full_weights[names[5]],
        ]
    ).cuda()
    torch.testing.assert_close(merged.mm_param.weight, expected_weight)

    x = torch.arange(6, dtype=torch.float32, device="cuda").view(2, 3)
    torch.testing.assert_close(merged.mm(x, use_custom_tensor_mananger=False), x @ expected_weight.T)
    assert merged.verify_load()


@pytest.mark.parametrize("shape", [(17, 128), (3, 32, 128)])
def test_fused_hadamard_fp8_quant_matches_two_kernel_chain(shape):
    torch.manual_seed(1)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected_value, expected_scale = act_quant(
        hadamard_transform(value, scale=128 ** -0.5),
        block_size=128,
        scale_fmt="ue8m0",
    )

    actual_value, actual_scale = hadamard_transform_quant_fp8(value, scale=128 ** -0.5)

    assert torch.equal(actual_value, expected_value)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


def test_fused_indexer_weight_scale_matches_torch_chain():
    torch.manual_seed(2)
    weights = torch.randn(257, 32, device="cuda", dtype=torch.float32)
    q_scale = torch.rand(257, 32, 1, device="cuda", dtype=torch.float32)
    scale = 128 ** -0.5 * 32 ** -0.5
    expected = (weights * scale).unsqueeze(-1).mul(q_scale).squeeze(-1)

    actual = scale_indexer_weights_(weights.clone(), q_scale, scale)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
