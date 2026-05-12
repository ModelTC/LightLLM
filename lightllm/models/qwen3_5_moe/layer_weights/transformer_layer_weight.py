import torch
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight


class Qwen35MOETransformerLayerWeight(Qwen35TransformerLayerWeight):
    def load_hf_weights(self, weights):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        split_fused_expert_weights(weights, self.layer_num_, moe_intermediate_size)
        return super().load_hf_weights(weights)


def split_fused_expert_weights(weights: dict, layer_num: int, moe_intermediate_size: int):
    layer_prefix = f"model.layers.{layer_num}."
    keys = list(weights.keys())

    for k in keys:
        if not k.startswith(layer_prefix):
            continue

        if "mlp.experts.gate_up_proj" in k:
            fused_weight = weights.pop(k)  # [num_experts, 2*inter_size, hidden_size]
            prefix = k.rsplit(".gate_up_proj", 1)[0]
            gate_weight = fused_weight[:, :moe_intermediate_size, :]
            up_weight = fused_weight[:, moe_intermediate_size:, :]
            weights[f"{prefix}.gate_proj"] = gate_weight
            weights[f"{prefix}.up_proj"] = up_weight
