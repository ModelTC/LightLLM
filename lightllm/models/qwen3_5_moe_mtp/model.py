from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.qwen3_5_moe_mtp.layer_weights.transformer_layer_weight import (
    Qwen3_5MoeMTPTransformerLayerWeight,
)


class Qwen3_5MoeMTPModel(Qwen3_5MTPModel):
    transformer_weight_class = Qwen3_5MoeMTPTransformerLayerWeight
