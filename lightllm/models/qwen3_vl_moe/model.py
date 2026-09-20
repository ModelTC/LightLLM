from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight
from lightllm.models.qwen3_vl_moe.layer_weights.transformers_layer_weight import Qwen3VLMOETransformerLayerWeight
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo


class Qwen3VLMOETpPartModel(Qwen3MOEModel):

    pre_layer_infer_class = Qwen3VLMultimodalPreLayerInfer
    transformer_layer_infer_class = Qwen3VLMOETransformerLayerInfer

    pre_and_post_weight_class = Qwen3VLPreAndPostLayerWeight
    transformer_weight_class = Qwen3VLMOETransformerLayerWeight

    infer_state_class = Qwen3VLInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        all_config = self._load_model_config_dict()
        self.config = all_config["text_config"]
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size
        return
