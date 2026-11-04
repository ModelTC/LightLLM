from lightllm.models.deepseek2.adapter import Deepseek2ModelAdapter
from lightllm.models.glm4_moe_lite.layer_infer.transformer_layer_infer import Glm4MoeLiteTransformerLayerInfer
from lightllm.models.glm4_moe_lite.layer_weights.transformer_layer_weight import Glm4MoeLiteTransformerLayerWeight


class Glm4MoeLiteModelAdapter(Deepseek2ModelAdapter):
    moe_group_score_topk = 2
    transformer_weight_class = Glm4MoeLiteTransformerLayerWeight
    transformer_layer_infer_class = Glm4MoeLiteTransformerLayerInfer
