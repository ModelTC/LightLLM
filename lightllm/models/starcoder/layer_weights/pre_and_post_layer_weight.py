import torch
import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight


class StarcoderPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):

        vob_size = self.network_config_["vocab_size"]
        split_vob_size = vob_size // self.world_size_
        n_embed = self.network_config_["hidden_size"]
        if "transformer.wte.weight" in weights:
            # print(weights['transformer.wte.weight'].shape)
            self.wte_weight_ = weights['transformer.wte.weight'][split_vob_size *
                                                                    self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if "transformer.wpe.weight" in weights:
            # print(weights['transformer.wpe.weight'].shape)
            self.wpe_weight_ = weights['transformer.wpe.weight'].to(self.data_type_).cuda()
        if 'lm_head.weight' in weights:
            self.lm_head_weight_ = weights['lm_head.weight'][split_vob_size * self.tp_rank_: split_vob_size *
                                                            (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if "transformer.ln_f.weight" in weights:
            self.final_norm_weight_ = weights['transformer.ln_f.weight'].contiguous().to(self.data_type_).cuda()
        if "transformer.ln_f.bias" in weights:
            self.final_norm_bias_ = weights["transformer.ln_f.bias"].contiguous().to(self.data_type_).cuda()
        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.final_norm_weight_, 
                   self.final_norm_bias_,
                   self.wte_weight_,
                   self.wpe_weight_,
                   self.lm_head_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 