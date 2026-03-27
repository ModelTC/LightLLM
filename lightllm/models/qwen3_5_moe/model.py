from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.utils.log_utils import init_logger
from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)


@ModelRegistry("qwen3_5_moe", is_multimodal=True)
class Qwen3_5MOETpPartModel(Qwen3_5TpPartModel):

    transformer_weight_class = Qwen35MOETransformerLayerWeight
