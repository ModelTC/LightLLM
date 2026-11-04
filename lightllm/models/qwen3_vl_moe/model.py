from lightllm.models.qwen3_vl_moe.adapter import Qwen3VLMoEModelAdapter
from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_moe.adapter import Qwen3MoEModelAdapter
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen3_vl_moe.layer_weights.transformers_layer_weight import Qwen3VLMOETransformerLayerWeight
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo


class Qwen3VLMOETpPartModel(Qwen3MOEModel):
    model_adapter_class = Qwen3VLMoEModelAdapter
    pre_layer_infer_class = Qwen3VLMoEModelAdapter.pre_layer_infer_class
    transformer_layer_infer_class = Qwen3VLMoEModelAdapter.transformer_layer_infer_class
    pre_and_post_weight_class = Qwen3VLMoEModelAdapter.pre_and_post_weight_class
    transformer_weight_class = Qwen3VLMoEModelAdapter.transformer_weight_class
    infer_state_class = Qwen3VLInferStateInfo

    def _init_config(self):
        TpPartBaseModel._init_config(self)
