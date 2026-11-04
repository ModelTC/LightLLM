from lightllm.models.qwen3_omni_moe_thinker.layer_infer.transformer_layer_infer import Qwen3OmniMOETransformerLayerInfer
from lightllm.models.qwen3_omni_moe_thinker.layer_weights.pre_and_post_layer_weight import (
    Qwen3OmniMOEThinkerPreAndPostLayerWeight,
)
from lightllm.models.qwen3_omni_moe_thinker.layer_weights.transformers_layer_weight import (
    Qwen3OmniMOEThinkerTransformerLayerWeight,
)
from lightllm.models.qwen3_vl_moe.adapter import Qwen3VLMoEModelAdapter


class Qwen3OmniMoEModelAdapter(Qwen3VLMoEModelAdapter):
    transformer_layer_infer_class = Qwen3OmniMOETransformerLayerInfer
    pre_and_post_weight_class = Qwen3OmniMOEThinkerPreAndPostLayerWeight
    transformer_weight_class = Qwen3OmniMOEThinkerTransformerLayerWeight
