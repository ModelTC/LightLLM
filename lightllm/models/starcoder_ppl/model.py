import os
import json
import torch

from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_ppl.layer_infer.transformer_layer_infer import StarcoderPPlTransformerLayerInfer
from lightllm.common.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager

class StarcoderPPlTpPartModel(StarcoderTpPartModel):

    # infer class
    transformer_layer_infer_class = StarcoderPPlTransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[]):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
    
    def _verify_params(self):
        assert self.load_way == "HF", "StarCoder only support HF format to load Now!"
        assert "int8kv" in self.mode, "ppl Starcoder only support int8kv mode"
        return

    def _init_mem_manager(self):
        self.mem_manager = PPLINT8KVMemoryManager(self.max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_key_value_heads"],
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])
        return