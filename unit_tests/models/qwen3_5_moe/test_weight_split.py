import torch

from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    split_fused_expert_weights,
)


def test_split_independent_expert_packed_weights_from_rl_update():
    num_experts = 3
    intermediate_size = 2
    hidden_size = 4
    prefix = "model.layers.7.mlp.experts"
    gate = torch.arange(num_experts * intermediate_size * hidden_size).view(num_experts, intermediate_size, hidden_size)
    up = gate + 100
    down = torch.arange(num_experts * hidden_size * intermediate_size).view(num_experts, hidden_size, intermediate_size)
    weights = {
        f"{prefix}.gate_proj.weight": gate,
        f"{prefix}.up_proj.weight": up,
        f"{prefix}.down_proj.weight": down,
        "model.layers.8.mlp.experts.gate_proj.weight": torch.empty(0),
    }

    split_fused_expert_weights(weights, layer_num=7, moe_intermediate_size=intermediate_size)

    assert f"{prefix}.gate_proj.weight" not in weights
    assert f"{prefix}.up_proj.weight" not in weights
    assert f"{prefix}.down_proj.weight" not in weights
    assert "model.layers.8.mlp.experts.gate_proj.weight" in weights
    for expert_idx in range(num_experts):
        torch.testing.assert_close(weights[f"{prefix}.{expert_idx}.gate_proj.weight"], gate[expert_idx])
        torch.testing.assert_close(weights[f"{prefix}.{expert_idx}.up_proj.weight"], up[expert_idx])
        torch.testing.assert_close(weights[f"{prefix}.{expert_idx}.down_proj.weight"], down[expert_idx])


def test_split_checkpoint_gate_up_expert_packed_weight():
    num_experts = 3
    intermediate_size = 2
    hidden_size = 4
    prefix = "model.layers.7.mlp.experts"
    gate = torch.arange(num_experts * intermediate_size * hidden_size).view(num_experts, intermediate_size, hidden_size)
    up = gate + 100
    gate_up = torch.cat((gate, up), dim=1)
    weights = {f"{prefix}.gate_up_proj": gate_up}

    split_fused_expert_weights(weights, layer_num=7, moe_intermediate_size=intermediate_size)

    assert f"{prefix}.gate_up_proj" not in weights
    for expert_idx in range(num_experts):
        torch.testing.assert_close(weights[f"{prefix}.{expert_idx}.gate_proj.weight"], gate[expert_idx])
        torch.testing.assert_close(weights[f"{prefix}.{expert_idx}.up_proj.weight"], up[expert_idx])
