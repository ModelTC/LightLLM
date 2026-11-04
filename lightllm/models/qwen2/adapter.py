from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight
from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight
from lightllm.models.llama.adapter import LlamaModelAdapter


class Qwen2ModelAdapter(LlamaModelAdapter):
    pre_and_post_weight_class = Qwen2PreAndPostLayerWeight
    transformer_weight_class = Qwen2TransformerLayerWeight
