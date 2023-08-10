import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.base_layer_weight import BaseLayerWeight


class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=""):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.q_weight_ = None
        self.k_weight_ = None
        self.v_weight_ = None
        self.att_out_dense_weight_ = None
        self.inv_freq = None
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self.input_layernorm = None
        self.post_attention_layernorm_weight_ = None
        self.mode = mode
        self.network_config = network_config

    def load_hf_weights(self, weights):
        # input layernorm params
        if f"transformer.h.{self.layer_num_}.ln_1.weight" in weights:
            self.input_layernorm = weights[f"transformer.h.{self.layer_num_}.ln_1.weight"].to(self.data_type_).cuda()

        # attention params
        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        if f"transformer.h.{self.layer_num_}.attn.c_attn.weight" in weights:
            qkv_weights = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"]
            split_size = qkv_weights.shape[0] // 3
            q_weights, k_weights, v_weights = torch.split(qkv_weights, split_size, dim=0)

            self.q_weight_ = q_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self.q_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.q_weight_ = self.q_weight_.cuda()
            self.k_weight_ = k_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self.k_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.k_weight_ = self.k_weight_.cuda()
            self.v_weight_ = v_weights[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self.v_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.v_weight_ = self.v_weight_.cuda()
        
        if f"transformer.h.{self.layer_num_}.attn.c_attn.bias" in weights:
            qkv_bias = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"]
            split_size = qkv_bias.shape[0] // 3
            q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size, dim=0)
            self.q_bias_ = q_bias.contiguous().to(self.data_type_).cuda()
            self.k_bias_ = k_bias.contiguous().to(self.data_type_).cuda()
            self.v_bias_ = v_bias.contiguous().to(self.data_type_).cuda()

        # attention output dense params
        if f"transformer.h.{self.layer_num_}.attn.c_proj.weight" in weights:
            self.att_out_dense_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.att_out_dense_weight_ = self.att_out_dense_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.att_out_dense_weight_ = self.att_out_dense_weight_.cuda()

        if f"transformer.h.{self.layer_num_}.ln_2.weight" in weights:
            self.post_attention_layernorm_weight_ = weights[f"transformer.h.{self.layer_num_}.ln_2.weight"].to(
                self.data_type_).cuda()

        # ffn params
        inter_size = self.network_config['ffn_hidden_size'] // 2
        split_inter_size = inter_size // self.world_size_

        if f"transformer.h.{self.layer_num_}.mlp.w1.weight" in weights:
            self.up_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w1.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = self.up_proj.transpose(0, 1).contiguous().to(self.data_type_)
            self.up_proj = self.up_proj.cuda()

        if f"transformer.h.{self.layer_num_}.mlp.w2.weight" in weights:
            self.gate_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w2.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = self.gate_proj.transpose(0, 1).contiguous().to(self.data_type_)

            self.gate_proj = self.gate_proj.cuda()


        if f"transformer.h.{self.layer_num_}.mlp.c_proj.weight" in weights:
            self.down_proj = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self.down_proj.transpose(0, 1).contiguous().to(self.data_type_)
            
            self.down_proj = self.down_proj.cuda()

        return
