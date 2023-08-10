import torch
import numpy as np
from lightllm.models.llama.layer_weights.base_layer_weight import BaseLayerWeight


class PreAndPostLayerWeight(BaseLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config):
        self.tp_rank_ = tp_rank
        self.data_type_ = data_type
        self.world_size_ = world_size
        self.final_layernorm_weight_ = None
        self.wte_weight_ = None
        self.lm_head_weight = None
        self.network_config = network_config

    def load_hf_weights(self, weights):

        vob_size = self.network_config["vocab_size"]
        split_vob_size = vob_size // self.world_size_

        if "transformer.wte.weight" in weights:
            self.wte_weight_ = weights['transformer.wte.weight'][split_vob_size *
                                                                    self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if 'lm_head.weight' in weights:
            self.lm_head_weight = weights['lm_head.weight'][split_vob_size * self.tp_rank_: split_vob_size *
                                                            (self.tp_rank_ + 1), :].contiguous().to(self.data_type_).cuda()
        if 'transformer.ln_f.weight' in weights:
            self.final_layernorm_weight_ = weights['transformer.ln_f.weight'].to(self.data_type_).cuda()

        return
