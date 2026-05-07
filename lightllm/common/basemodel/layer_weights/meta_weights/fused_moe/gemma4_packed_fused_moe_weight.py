from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight


class Gemma4PackedFusedMoeWeight(FusedMoeWeight):
    def load_hf_weights(self, weights):
        gate_up_name = f"{self.weight_prefix}.gate_up_proj"
        down_name = f"{self.weight_prefix}.down_proj"
        if gate_up_name not in weights and down_name not in weights and self.per_expert_scale_name not in weights:
            return super().load_hf_weights(weights)

        assert self.quant_method.method_name == "none", "Gemma-4 packed MoE currently supports bf16/no-quant weights."
        assert not self.enable_ep_moe, "Gemma-4 packed MoE currently supports TP mode only."

        start = self.split_inter_size * self.tp_rank_
        end = self.split_inter_size * (self.tp_rank_ + 1)
        moe_intermediate_size = self.moe_intermediate_size

        if gate_up_name in weights:
            gate_up_weight = weights[gate_up_name]
            for expert_idx, local_expert_idx in self.expert_idx_to_local_idx.items():
                gate_weight = gate_up_weight[expert_idx, start:end, :].contiguous()
                up_weight = gate_up_weight[
                    expert_idx, moe_intermediate_size + start : moe_intermediate_size + end, :
                ].contiguous()
                self.quant_method.load_weight(gate_weight, self.w1_list[local_expert_idx])
                self.quant_method.load_weight(up_weight, self.w3_list[local_expert_idx])

        if down_name in weights:
            down_weight = weights[down_name]
            for expert_idx, local_expert_idx in self.expert_idx_to_local_idx.items():
                down_weight_slice = down_weight[expert_idx, :, start:end].contiguous()
                self.quant_method.load_weight(down_weight_slice, self.w2_list[local_expert_idx])

        self._load_per_expert_scale(weights)
