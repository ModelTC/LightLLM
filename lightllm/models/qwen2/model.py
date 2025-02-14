import os
import json
import torch

from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight
from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight

from lightllm.models.llama.model import LlamaTpPartModel


class Qwen2TpPartModel(LlamaTpPartModel):
    # weight class
    pre_and_post_weight_class = Qwen2PreAndPostLayerWeight
    transformer_weight_class = Qwen2TransformerLayerWeight

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        if self.config["sliding_window"] is None:
            self.config["sliding_window"] = self.max_total_token_num
        # rename key [SYM: to be confirmed]
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "mistral only supports HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
