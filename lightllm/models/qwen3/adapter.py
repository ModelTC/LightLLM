from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.qwen2.adapter import Qwen2ModelAdapter


class Qwen3ModelAdapter(Qwen2ModelAdapter):
    transformer_weight_class = Qwen3TransformerLayerWeight

    transformer_layer_infer_class = Qwen3TransformerLayerInfer
