from lightllm.models.qwen3_5_moe_mtp.adapter import Qwen35MoeMTPModelAdapter
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.qwen3_5_mtp.adapter import Qwen35MTPModelAdapter
from lightllm.models.qwen3_5_moe.adapter import Qwen35MoEModelAdapter
from lightllm.models.qwen3_5_moe_mtp.layer_weights.transformer_layer_weight import (
    Qwen3_5MoeMTPTransformerLayerWeight,
)


class Qwen3_5MoeMTPModel(Qwen3_5MTPModel):
    model_adapter_class = Qwen35MoeMTPModelAdapter
    transformer_weight_class = Qwen3_5MoeMTPTransformerLayerWeight
