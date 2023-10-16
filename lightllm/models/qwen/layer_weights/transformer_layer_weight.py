import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight


class QwenTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):
        # input layernorm params
        if f"transformer.h.{self.layer_num_}.ln_1.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_1.weight"])

        # attention params
        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        if f"transformer.h.{self.layer_num_}.attn.c_attn.weight" in weights:
            qkv_weights = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"]
            split_size = qkv_weights.shape[0] // 3
            q_weights, k_weights, v_weights = torch.split(qkv_weights, split_size, dim=0)

            self.q_weight_ = q_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))
            self.k_weight_ = k_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(self.k_weight_.transpose(0, 1))
            self.v_weight_ = v_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(self.v_weight_.transpose(0, 1))
            self.v_weight_ = self.v_weight_.cuda()
        
        if f"transformer.h.{self.layer_num_}.attn.c_attn.bias" in weights:
            qkv_bias = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"]
            split_size = qkv_bias.shape[0] // 3
            q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size, dim=0)
            self.q_bias_ = self._cuda(q_bias[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])
            self.k_bias_ = self._cuda(k_bias[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])
            self.v_bias_ = self._cuda(v_bias[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)])

        # attention output dense params
        if f"transformer.h.{self.layer_num_}.attn.c_proj.weight" in weights:
            self.o_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))

        if f"transformer.h.{self.layer_num_}.ln_2.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_2.weight"])
        # ffn params
        inter_size = self.network_config_['ffn_hidden_size'] // 2
        split_inter_size = inter_size // self.world_size_

        if f"transformer.h.{self.layer_num_}.mlp.w1.weight" in weights:
            self.up_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w1.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = self._cuda(self.up_proj.transpose(0, 1))

        if f"transformer.h.{self.layer_num_}.mlp.w2.weight" in weights:
            self.gate_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w2.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = self._cuda(self.gate_proj.transpose(0, 1))

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.weight" in weights:
            self.down_proj = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))
            
        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.q_weight_,
                   self.k_weight_,
                   self.v_weight_,
                   self.q_bias_,
                   self.k_bias_,
                   self.v_bias_,
                   self.o_weight_,
                   self.ffn_norm_weight_,
                   self.up_proj,
                   self.gate_proj,
                   self.down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 