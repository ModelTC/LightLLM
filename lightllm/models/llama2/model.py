import os
import json
import torch

from lightllm.models.llama2.layer_infer.transformer_layer_infer import Llama2TransformerLayerInfer
from lightllm.models.llama2.layer_weights.transformer_layer_weight import Llama2TransformerLayerWeight

from lightllm.common.mem_manager import MemoryManager
from lightllm.models.llama.model import LlamaTpPartModel


class Llama2TpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Llama2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Llama2TransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
    

    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        return 
    
    def _verify_params(self):
        assert self.load_way == "HF", "llama only support HF format to load Now!"
        assert self.mode == "", "future to support int8 int4 ..."
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return
    
    def _init_mem_manager(self):
        mem_dict = {
            "" : MemoryManager,
        }
        
        self.mem_manager = mem_dict[self.mode](self.max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_key_value_heads"] // self.world_size_,
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])
        return
    
    def _init_some_value(self):
        self.head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return