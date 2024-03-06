import os
import json
import torch

from lightllm.models.internlm2.layer_weights.pre_and_post_layer_weight import Internlm2PreAndPostLayerWeight
from lightllm.models.internlm2_wquant.layer_weights.transformer_layer_weight import Internlm2TransformerLayerWeightQuantized
from lightllm.models.internlm_wquant.model import InternlmTpPartModelWQuant
from lightllm.common.mem_utils import select_mem_manager_class


class Internlm2TpPartModelWQuant(InternlmTpPartModelWQuant):
    # weight class
    pre_and_post_weight_class = Internlm2PreAndPostLayerWeight 
    transformer_weight_class = Internlm2TransformerLayerWeightQuantized

    def __init__(self, kvargs):
        super().__init__(kvargs)
    
    def _verify_params(self):
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert any("w4a16" in mode_ or "w8a16" in mode_ for mode_ in self.mode), "only for weight quant model"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode)(self.max_total_token_num, 
                                                     dtype=torch.float16,
                                                     head_num=self.config["num_key_value_heads"] // self.world_size_,
                                                     head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                                     layer_num=self.config["num_hidden_layers"],
                                                     always_copy=True)
        return