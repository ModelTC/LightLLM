import os
import json
import torch

from lightllm.models.chatglm2.layer_infer.transformer_layer_infer import ChatGLM2TransformerLayerInfer
from lightllm.models.chatglm2.layer_weights.transformer_layer_weight import ChatGLM2TransformerLayerWeight
from lightllm.models.chatglm2.layer_weights.pre_and_post_layer_weight import ChatGLM2PreAndPostLayerWeight

from lightllm.common.mem_manager import MemoryManager
from lightllm.models.llama2.model import Llama2TpPartModel
from lightllm.common.build_utils import repair_config


class ChatGlm2TpPartModel(Llama2TpPartModel):
    # weight class
    pre_and_post_weight_class = ChatGLM2PreAndPostLayerWeight
    transformer_weight_class = ChatGLM2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = ChatGLM2TransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer", "num_layers"])
        repair_config(self.config, same_names=["vocab_size", "padded_vocab_size"])
        repair_config(self.config, same_names=["rms_norm_eps", "layernorm_epsilon"])
        repair_config(self.config, same_names=["multi_query_group_num", "num_key_value_heads"])
        return 
    
    def _verify_params(self):
        assert self.load_way == "HF", "chatGLM2 only support HF format to load Now!"
        assert self.mode == "", "future to support int8 int4 ..."
        return
    
    def _init_mem_manager(self):
        mem_dict = {
            "" : MemoryManager,
        }
        
        self.mem_manager = mem_dict[self.mode](self.max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["multi_query_group_num"],
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_key_head_num_ = self.config["multi_query_group_num"]
        self.tp_value_head_num_ = self.config["multi_query_group_num"]