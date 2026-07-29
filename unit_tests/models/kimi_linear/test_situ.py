import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.triton_impl import FuseMoeTriton
from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.quantization.quantize_method import WeightPack


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SiTU is implemented by a Triton CUDA kernel")
@pytest.mark.parametrize("linear_beta", [None, 25.0])
def test_situ_and_mul_matches_kimi_k3_definition(linear_beta):
    torch.manual_seed(17)
    gate, up = torch.randn((11, 64), device="cuda", dtype=torch.float32).chunk(2, dim=-1)
    packed = torch.cat((gate, up), dim=-1).contiguous()
    actual = torch.empty_like(gate)

    silu_and_mul_fwd(
        packed,
        actual,
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=linear_beta,
        run_config={"BLOCK_M": 1, "BLOCK_N": 32, "num_warps": 1, "NUM_STAGES": 1},
    )

    expected_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_up = up if linear_beta is None else linear_beta * torch.tanh(up / linear_beta)
    torch.testing.assert_close(actual, expected_gate * expected_up, rtol=1e-5, atol=1e-6)


def test_fused_moe_forwards_situ_parameters(monkeypatch):
    captured = {}

    def fake_fused_experts_impl(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(grouped_fused_moe, "fused_experts_impl", fake_fused_experts_impl)
    impl = object.__new__(FuseMoeTriton)
    impl.activation = "situ"
    impl.activation_situ_beta = 4.0
    impl.activation_situ_linear_beta = 25.0
    hidden_states = torch.zeros((2, 4))
    weight = WeightPack(weight=torch.zeros((3, 8, 4)))

    output = impl._fused_experts(
        hidden_states,
        w13=weight,
        w2=weight,
        topk_weights=torch.ones((2, 1)),
        topk_ids=torch.zeros((2, 1), dtype=torch.int64),
    )

    assert output is hidden_states
    assert captured["activation"] == "situ"
    assert captured["activation_situ_beta"] == 4.0
    assert captured["activation_situ_linear_beta"] == 25.0
