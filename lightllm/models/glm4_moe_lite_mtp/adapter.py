from lightllm.models.deepseek_mtp.adapter import Deepseek3MTPModelAdapter
from lightllm.models.glm4_moe_lite.layer_weights.transformer_layer_weight import Glm4MoeLiteTransformerLayerWeight
from lightllm.models.glm4_moe_lite.layer_infer.transformer_layer_infer import Glm4MoeLiteTransformerLayerInfer
from lightllm.models.glm4_moe_lite_mtp.layer_weights.pre_and_post_layer_weight import (
    Glm4MoeLiteMTPPreAndPostLayerWeight,
)


class Glm4MoeLiteMTPModelAdapter(Deepseek3MTPModelAdapter):
    pre_and_post_weight_class = Glm4MoeLiteMTPPreAndPostLayerWeight
    transformer_weight_class = Glm4MoeLiteTransformerLayerWeight
    transformer_layer_infer_class = Glm4MoeLiteTransformerLayerInfer
