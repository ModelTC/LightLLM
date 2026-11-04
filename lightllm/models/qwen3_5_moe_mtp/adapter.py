from lightllm.models.qwen3_5_mtp.adapter import Qwen35MTPModelAdapter
from lightllm.models.qwen3_5_moe.adapter import Qwen35MoEModelAdapter
from lightllm.models.qwen3_5_moe_mtp.layer_weights.transformer_layer_weight import Qwen3_5MoeMTPTransformerLayerWeight


class Qwen35MoeMTPModelAdapter(Qwen35MTPModelAdapter):
    transformer_weight_class = Qwen3_5MoeMTPTransformerLayerWeight
    moe_specs = Qwen35MoEModelAdapter.moe_specs
