from lightllm.models.llama.adapter import LlamaModelAdapter
from lightllm.models.qwen2.adapter import Qwen2ModelAdapter
from lightllm.models.qwen2_vl.adapter import Qwen2VLModelAdapter
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.tarsier2.layer_weights.pre_and_post_layer_weight import (
    Tarsier2Qwen2PreAndPostLayerWeight,
    Tarsier2LlamaPreAndPostLayerWeight,
)


class Tarsier2Qwen2ModelAdapter(Qwen2ModelAdapter):
    pre_and_post_weight_class = Tarsier2Qwen2PreAndPostLayerWeight
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer


class Tarsier2Qwen2VLModelAdapter(Qwen2VLModelAdapter):
    pre_and_post_weight_class = Tarsier2Qwen2PreAndPostLayerWeight
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer


class Tarsier2LlamaModelAdapter(LlamaModelAdapter):
    pre_and_post_weight_class = Tarsier2LlamaPreAndPostLayerWeight
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
