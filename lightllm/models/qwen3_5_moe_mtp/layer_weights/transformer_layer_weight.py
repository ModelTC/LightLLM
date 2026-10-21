from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)
from lightllm.models.qwen3_5_mtp.layer_weights.transformer_layer_weight import rename_mtp_weight_keys


class Qwen3_5MoeMTPTransformerLayerWeight(Qwen35MOETransformerLayerWeight):
    def load_hf_weights(self, weights):
        rename_mtp_weight_keys(weights)
        return super().load_hf_weights(weights)
