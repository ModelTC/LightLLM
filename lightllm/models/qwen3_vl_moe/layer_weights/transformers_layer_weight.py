from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight


class Qwen3VLMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def load_hf_weights(self, weights):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        moe_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        gate_up_name = f"{moe_prefix}.gate_up_proj"
        down_name = f"{moe_prefix}.down_proj"

        if gate_up_name in weights:
            gate_up = weights.pop(gate_up_name)
            gate_weight = gate_up[:, :, :moe_intermediate_size].transpose(1, 2).contiguous()
            up_weight = gate_up[:, :, moe_intermediate_size:].transpose(1, 2).contiguous()

            for expert_idx in range(gate_up.shape[0]):
                weights[f"{moe_prefix}.{expert_idx}.gate_proj.weight"] = gate_weight[expert_idx]
                weights[f"{moe_prefix}.{expert_idx}.up_proj.weight"] = up_weight[expert_idx]

        if down_name in weights:
            down_weight = weights.pop(down_name)
            for expert_idx in range(down_weight.shape[0]):
                weights[f"{moe_prefix}.{expert_idx}.down_proj.weight"] = (
                    down_weight[expert_idx].transpose(0, 1).contiguous()
                )

        return super().load_hf_weights(weights)
