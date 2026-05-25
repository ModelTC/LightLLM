from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight


class Gemma4PackedFusedMoeWeight(FusedMoeWeight):
    def load_hf_weights(self, weights):
        # 将权重名称的格式对齐基类的统一加载格式。
        gate_up_name = f"{self.weight_prefix}.gate_up_proj"
        down_name = f"{self.weight_prefix}.down_proj"
        assert not self.enable_ep_moe, "Gemma-4 packed MoE currently supports TP mode only."
        moe_intermediate_size = self.moe_intermediate_size

        if gate_up_name in weights:
            gate_up_weight = weights[gate_up_name]

            for expert_idx in range(self.n_routed_experts):
                expert_gate_weight = gate_up_weight[expert_idx, :moe_intermediate_size, :]
                expert_up_weight = gate_up_weight[expert_idx, moe_intermediate_size:, :]

                weights[
                    f"{self.weight_prefix}.{expert_idx}.{self.w1_weight_name}.{self.quant_method.weight_suffix}"
                ] = expert_gate_weight
                weights[
                    f"{self.weight_prefix}.{expert_idx}.{self.w3_weight_name}.{self.quant_method.weight_suffix}"
                ] = expert_up_weight

            del weights[gate_up_name]

        if down_name in weights:
            down_weight = weights[down_name]
            for expert_idx in range(self.n_routed_experts):
                expert_down_weight = down_weight[expert_idx, :, :]
                weights[
                    f"{self.weight_prefix}.{expert_idx}.{self.w2_weight_name}.{self.quant_method.weight_suffix}"
                ] = expert_down_weight
            del weights[down_name]

        super().load_hf_weights(weights)
        return
