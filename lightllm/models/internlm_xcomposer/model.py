from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer
from lightllm.models.internlm_xcomposer.layer_infer.transformer_layer_infer import InternlmComposerTransformerLayerInfer
from lightllm.models.internlm_xcomposer.layer_weights.transformer_layer_weight import InternlmComposerTransformerLayerWeight
from lightllm.models.internlm_xcomposer.infer_struct import InternlmComposerInferStateInfo


class InternlmComposerTpPartModel(Internlm2TpPartModel):
    transformer_weight_class = InternlmComposerTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaMultimodalPreLayerInfer
    transformer_layer_infer_class = InternlmComposerTransformerLayerInfer

    # infer state class
    infer_state_class = InternlmComposerInferStateInfo
    
    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
