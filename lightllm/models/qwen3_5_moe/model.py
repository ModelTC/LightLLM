from lightllm.models.qwen3_5_moe.adapter import Qwen35MoEModelAdapter
from dataclasses import replace

from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5.adapter import Qwen35ModelAdapter
from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import (
    Qwen35MOETransformerLayerWeight,
)


class Qwen3_5MOETpPartModel(Qwen3_5TpPartModel):
    model_adapter_class = Qwen35MoEModelAdapter
    transformer_weight_class = Qwen35MOETransformerLayerWeight
