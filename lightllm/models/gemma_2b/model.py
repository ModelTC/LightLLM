from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import Gemma_2bTransformerLayerWeight
from lightllm.models.gemma_2b.layer_weights.pre_and_post_layer_weight import Gemma_2bPreAndPostLayerWeight
from lightllm.models.gemma_2b.layer_infer.pre_layer_infer import Gemma_2bPreLayerInfer
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import Gemma_2bTransformerLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel


class Gemma_2bTpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = Gemma_2bPreAndPostLayerWeight
    transformer_weight_class = Gemma_2bTransformerLayerWeight

    # infer class
    pre_layer_infer_class = Gemma_2bPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = Gemma_2bTransformerLayerInfer

    # infer state class
    infer_state_class = LlamaInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "gemma only supports HF format to load Now!"
        # assert self.config["num_key_value_heads"] % self.tp_world_size_ == 0
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    @classmethod
    def _full_attention_options(cls, config, args):
        from lightllm.common.basemodel.basemodel_utils import get_full_attention_options

        options = get_full_attention_options(config, args)
        options.update(dim=config["hidden_size"] // config["num_attention_heads"], replicated=True)
        return options
