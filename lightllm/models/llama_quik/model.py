import os
import json
from functools import partial
import torch

from lightllm.models.llama_quik.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeightQuik
from lightllm.models.llama_quik.layer_infer.transformer_layer_infer import LlamaTransformerLayerInferQuik
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.mem_utils import select_mem_manager_class


class LlamaTpPartModelQuik(LlamaTpPartModel):
    # weight class
    transformer_weight_class = LlamaTransformerLayerWeightQuik

    # infer class
    transformer_layer_infer_class = LlamaTransformerLayerInferQuik

    def __init__(self, kvargs):
        super().__init__(kvargs)

    def _verify_params(self):
        assert self.load_way in ["HF", "DS"], "llama only supports HF and DS format to load Now!"
        assert any("quik_activation_weight" in mode_ for mode_ in self.mode), "only for quik quant model"
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
        