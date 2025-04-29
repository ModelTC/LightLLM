import torch
from typing import final
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_utils import select_mem_manager_class


logger = init_logger(__name__)


class Qwen3TpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Qwen3TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3TransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
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
