from lightllm.models.qwen3.layer_weights.transformer_layer_weight import (
    Qwen3TransformerLayerWeight,
)


class Cosmos3TextTransformerLayerWeight(Qwen3TransformerLayerWeight):
    """Map Cosmos3's unprefixed text-tower tensors onto Qwen3 weights."""

    def _init_weight_names(self):
        super()._init_weight_names()
        prefix = f"layers.{self.layer_num_}"
        attention = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        self._q_weight_name = f"{attention}.to_q.weight"
        self._k_weight_name = f"{attention}.to_k.weight"
        self._v_weight_name = f"{attention}.to_v.weight"
        self._o_weight_name = f"{attention}.to_out.weight"
        self._q_norm_name = f"{attention}.norm_q.weight"
        self._k_norm_name = f"{attention}.norm_k.weight"
        self._gate_weight_name = f"{mlp}.gate_proj.weight"
        self._up_weight_name = f"{mlp}.up_proj.weight"
        self._down_weight_name = f"{mlp}.down_proj.weight"
        self._att_norm_weight_name = f"{prefix}.input_layernorm.weight"
        self._ffn_norm_weight_name = f"{prefix}.post_attention_layernorm.weight"
