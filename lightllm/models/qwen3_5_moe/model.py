from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)
from lightllm.models.qwen3_5_moe.layer_infer.transformer_layer_infer import (
    Qwen35MOETransformerLayerInfer,
)


@ModelRegistry("qwen3_5_moe", is_multimodal=True)
class Qwen3_5MOETpPartModel(Qwen3_5TpPartModel):

    transformer_weight_class = Qwen35MOETransformerLayerWeight
    transformer_layer_infer_class = Qwen35MOETransformerLayerInfer
