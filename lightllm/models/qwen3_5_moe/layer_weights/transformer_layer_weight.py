from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight


class Qwen35MOETransformerLayerWeight(Qwen35TransformerLayerWeight):
    def load_hf_weights(self, weights):
        split_fused_expert_weights(weights, self.layer_num_, self.network_config_["moe_intermediate_size"])
        return super().load_hf_weights(weights)


def split_fused_expert_weights(weights: dict, layer_num: int, moe_intermediate_size: int):
    """将 HF 打包的 fused MoE expert 权重拆成按 expert 索引的独立权重。

    部分 checkpoint（如 Qwen3.5-MoE）把所有 expert 的 gate_up / down 压成
    ``mlp.experts.{gate_up,down}_proj`` 的打包张量（首维为 expert 数）。
    本函数只处理 ``model.layers.{layer_num}`` 下的这类 key：弹出打包权重，
    再写入 ``mlp.experts.{expert_idx}.{gate,up,down}_proj.weight``，供后续
    按 expert 加载。``gate_up_proj`` 会按 ``moe_intermediate_size`` 沿
    intermediate 维切成 gate / up。
    """
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
