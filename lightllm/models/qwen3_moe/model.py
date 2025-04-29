import torch
from typing import final
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.mem_utils import select_mem_manager_class
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class Qwen3MOEModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Qwen3MOETransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3MOETransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _verify_params(self):
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert self.config["num_attention_heads"] % self.tp_world_size_ == 0
        return

    def _init_some_value(self):
        # Dealing with head_dim_!=n_embed // num_attention_heads scenarios, such as mistral 13B
        head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.head_dim_ = self.config.get("head_dim", head_dim_)
        self.tp_k_head_num_ = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return

    def _init_mem_manager(self):
        head_dim_ = self.config["hidden_size"] // self.config["num_attention_heads"]
        head_dim_ = self.config.get("head_dim", head_dim_)
        tp_k_head_num_ = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)
        self.mem_manager = select_mem_manager_class(self.mode)(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=tp_k_head_num_,
            head_dim=head_dim_,
            layer_num=self.config["num_hidden_layers"],
            mem_fraction=self.mem_fraction,
        )
        return
