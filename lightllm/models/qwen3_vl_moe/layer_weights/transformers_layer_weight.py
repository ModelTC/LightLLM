from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight


class Qwen3VLMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def load_hf_weights(self, weights):
        self._align_fused_expert_weight_layout(weights)
        return super().load_hf_weights(weights)

    def _align_fused_expert_weight_layout(self, weights: dict) -> None:
        """将 Qwen3-VL-MoE 的 fused expert 权重布局对齐到基类期望格式。

        Qwen3-VL-MoE 官方 safetensor 中，expert 权重以 packed 3D 张量存储，但维度
        顺序与 Qwen3-MoE 文本模型 / ``nn.Linear`` 权重布局不同：

        - safetensor（VL）：
          - ``gate_up_proj``: ``[E, H, 2I]``
          - ``down_proj``: ``[E, I, H]``
        - 基类 ``split_fused_expert_weights`` 期望（与 Linear 一致）：
          - ``gate_up_proj``: ``[E, 2I, H]``
          - ``down_proj``: ``[E, H, I]``

        本函数仅对当前层存在的 fused key 做 ``transpose(1, 2)`` 布局转换，
        不负责按 expert 拆分；拆分仍复用基类逻辑。若对应 key 不在本批
        ``weights`` 中（分片加载），则跳过。
        """
        moe_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        gate_up_name = f"{moe_prefix}.gate_up_proj"
        down_name = f"{moe_prefix}.down_proj"

        if gate_up_name in weights:
            # [E, H, 2I] -> [E, 2I, H]
            weights[gate_up_name] = weights[gate_up_name].transpose(1, 2).contiguous()
        if down_name in weights:
            # [E, I, H] -> [E, H, I]
            weights[down_name] = weights[down_name].transpose(1, 2).contiguous()
