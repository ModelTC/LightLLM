from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_cohere_template import (
    TransformerLayerCohereInferTpl,
)
from lightllm.models.cohere.infer_struct import CohereInferStateInfo
from lightllm.models.cohere.layer_infer.post_layer_infer import CoherePostLayerInfer
from lightllm.models.cohere.layer_infer.transformer_layer_infer import CohereTransformerLayerInfer
from lightllm.models.cohere.layer_weights.pre_and_post_layer_weight import CoherePreAndPostLayerWeight
from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.cohere.splitfuse_infer_struct import CohereSplitFuseInferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.model import LlamaTpPartModel


class CohereTpPartModel(LlamaTpPartModel):
    pre_and_post_weight_class = CoherePreAndPostLayerWeight
    transformer_weight_class = CohereTransformerLayerWeight

    pre_layer_infer_class = LlamaPreLayerInfer
    transformer_layer_infer_class = CohereTransformerLayerInfer
    post_layer_infer_class = CoherePostLayerInfer

    infer_state_class = CohereInferStateInfo
    splitfuse_infer_state_class = CohereSplitFuseInferStateInfo
