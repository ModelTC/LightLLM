from lightllm.models.deepseek_v4_dspark.layer_infer.transformer_layer_infer import (
    DeepseekV4DSparkTransformerLayerInfer,
)
from lightllm.models.deepseek_v41.layer_infer.transformer_layer_infer import DeepseekV41TransformerLayerInfer


class DeepseekV41DSparkTransformerLayerInfer(DeepseekV4DSparkTransformerLayerInfer, DeepseekV41TransformerLayerInfer):
    """V4 KV-only context commits and V4.1 proposal forwards."""
