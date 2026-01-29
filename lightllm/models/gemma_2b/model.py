from lightllm.models.registry import ModelRegistry
from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import Gemma_2bTransformerLayerWeight
from lightllm.models.gemma_2b.layer_weights.pre_and_post_layer_weight import Gemma_2bPreAndPostLayerWeight
from lightllm.models.gemma_2b.layer_infer.pre_layer_infer import Gemma_2bPreLayerInfer
from lightllm.models.gemma_2b.layer_infer.transformer_layer_infer import Gemma_2bTransformerLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num


@ModelRegistry("gemma")
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

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class()(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_key_value_heads"],
            head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
            layer_num=self.config["num_hidden_layers"] + get_added_mtp_kv_layer_num(),
            mem_fraction=self.mem_fraction,
        )
        return
