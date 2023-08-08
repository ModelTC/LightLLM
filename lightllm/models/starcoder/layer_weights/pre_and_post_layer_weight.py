import torch
import numpy as np
from .base_layer_weight import BaseLayerWeight


class PreAndPostLayerWeight(BaseLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config):
        self.tp_rank_ = tp_rank
        self.data_type_ = data_type
        self.world_size_ = world_size
        self.wte_weight_ = None
        self.wpe_weight_ = None
        self.lm_head_weight = None
        self.final_layernorm_weight_ = None 
        self.final_layernorm_bias_ = None
        self.network_config = network_config

    def load_hf_weights(self, weights):
        # input layernorm params

        vob_size = self.network_config["vocab_size"]
        split_vob_size = vob_size // self.world_size_
        n_embed = self.network_config["hidden_size"]
        if "transformer.wte.weight" in weights:
            # print(weights['transformer.wte.weight'].shape)
            self.wte_weight_ = weights['transformer.wte.weight'][split_vob_size *
                                                                    self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if "transformer.wpe.weight" in weights:
            # print(weights['transformer.wpe.weight'].shape)
            self.wpe_weight_ = weights['transformer.wpe.weight'].to(self.data_type_).cuda()
        
        if 'lm_head.weight' in weights:
            # print(weights['lm_head.weight'].shape)
            self.lm_head_weight = weights['lm_head.weight'][split_vob_size * self.tp_rank_: split_vob_size *
                                                            (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if "transformer.ln_f.weight" in weights:
            self.final_layernorm_weight_ = weights['transformer.ln_f.weight'].contiguous().to(self.data_type_).cuda()
        if "transformer.ln_f.bias" in weights:
            self.final_layernorm_bias_ = weights["transformer.ln_f.bias"].contiguous().to(self.data_type_).cuda()
        return
