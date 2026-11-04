from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer
from lightllm.models.qwen2.adapter import Qwen2ModelAdapter


class Qwen2VLModelAdapter(Qwen2ModelAdapter):
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen2VLTransformerLayerInfer
