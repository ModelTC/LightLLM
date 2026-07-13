from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight


class Qwen35MOETransformerLayerWeight(Qwen35TransformerLayerWeight):
    def load_hf_weights(self, weights):
        split_fused_expert_weights(weights, self.layer_num_, self.network_config_["moe_intermediate_size"])
        return super().load_hf_weights(weights)


def split_fused_expert_weights(weights: dict, layer_num: int, moe_intermediate_size: int):
    layer_prefix = f"model.layers.{layer_num}."
    keys = list(weights.keys())

    for k in keys:
        if not k.startswith(layer_prefix):
            continue

        if "mlp.experts.gate_up_proj" in k:
            fused_weight = weights.pop(k)
            prefix = k.rsplit(".gate_up_proj", 1)[0]
            gate_weight = fused_weight[:, :moe_intermediate_size, :]
            up_weight = fused_weight[:, moe_intermediate_size:, :]

            for expert_idx in range(fused_weight.shape[0]):
                weights[f"{prefix}.{expert_idx}.gate_proj.weight"] = gate_weight[expert_idx]
                weights[f"{prefix}.{expert_idx}.up_proj.weight"] = up_weight[expert_idx]

        elif "mlp.experts.down_proj" in k:
            down_weight = weights.pop(k)
            prefix = k.rsplit(".down_proj", 1)[0]

            for expert_idx in range(down_weight.shape[0]):
                weights[f"{prefix}.{expert_idx}.down_proj.weight"] = down_weight[expert_idx]
