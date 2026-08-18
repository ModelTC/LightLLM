from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.draft_registry import DraftModelRegistry
from lightllm.models.qwen3_5_moe_mtp.layer_weights.transformer_layer_weight import (
    Qwen3_5MoeMTPTransformerLayerWeight,
)


@DraftModelRegistry(
    model_type=("qwen3_5_moe", "qwen3_5_moe_text"),
    spec_modes=("vanilla_with_att", "eagle_with_att"),
)
class Qwen3_5MoeMTPModel(Qwen3_5MTPModel):
    transformer_weight_class = Qwen3_5MoeMTPTransformerLayerWeight
