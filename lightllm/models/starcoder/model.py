import os
import json
import torch

from lightllm.models.starcoder.layer_infer.transformer_layer_infer import StarcoderTransformerLayerInfer
from lightllm.models.starcoder.layer_infer.pre_layer_infer import StarcoderPreLayerInfer
from lightllm.models.starcoder.layer_infer.infer_struct import StarcoderInferStateInfo
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight
from lightllm.models.starcoder.layer_weights.pre_and_post_layer_weight import StarcoderPreAndPostLayerWeight

from lightllm.common.mem_manager import MemoryManager
from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.common.build_utils import repair_config


class StarcoderTpPartModel(BloomTpPartModel):
    # weight class
    pre_and_post_weight_class = StarcoderPreAndPostLayerWeight
    transformer_weight_class = StarcoderTransformerLayerWeight

    # infer class
    pre_layer_infer_class = StarcoderPreLayerInfer
    transformer_layer_infer_class = StarcoderTransformerLayerInfer
    infer_state_class = StarcoderInferStateInfo

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode)
    
    def _init_config(self):
        super()._init_config()
        # rename key
        # repair_config()
        repair_config(self.config, same_names=["rms_norm_eps", "layer_norm_epsilon"])
        self.config['num_key_value_heads'] = 1
        return 
    
    def _verify_params(self):
        assert self.load_way == "HF", "StarCoder only support HF format to load Now!"
        assert self.mode == "", "future to support int8 int4 ..."
        return
    
    def _init_mem_manager(self):
        mem_dict = {
            "" : MemoryManager,
        }
        
        self.mem_manager = mem_dict[self.mode](self.max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_key_value_heads"],
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = self.config["num_key_value_heads"]
        self.tp_v_head_num_ = self.config["num_key_value_heads"]