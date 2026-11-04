from lightllm.models.llama.adapter import LlamaModelAdapter
from lightllm.models.qwen2.adapter import Qwen2ModelAdapter
from lightllm.models.deepseek2.adapter import Deepseek2ModelAdapter
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.internvl.layer_weights.pre_and_post_layer_weight import InternVLLlamaPreAndPostLayerWeight


class InternVLLlamaModelAdapter(LlamaModelAdapter):
    pre_and_post_weight_class = InternVLLlamaPreAndPostLayerWeight
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer


class InternVLQwen2ModelAdapter(Qwen2ModelAdapter):
    pre_and_post_weight_class = InternVLLlamaPreAndPostLayerWeight
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer


class InternVLDeepSeek2ModelAdapter(Deepseek2ModelAdapter):
    pre_and_post_weight_class = InternVLLlamaPreAndPostLayerWeight
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
