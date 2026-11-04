from lightllm.models.qwen3next.adapter import Qwen3NextModelAdapter
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_5.layer_weights.pre_and_post_layer_weight import Qwen35PreAndPostLayerWeight
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer


class Qwen35ModelAdapter(Qwen3NextModelAdapter):
    transformer_weight_class = Qwen35TransformerLayerWeight
    pre_and_post_weight_class = Qwen35PreAndPostLayerWeight
    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen35TransformerLayerInfer
