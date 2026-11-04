from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen3.adapter import Qwen3ModelAdapter


class Qwen3VLModelAdapter(Qwen3ModelAdapter):
    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen3VLTransformerLayerInfer
    pre_and_post_weight_class = Qwen3VLPreAndPostLayerWeight
