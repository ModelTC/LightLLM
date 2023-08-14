import torch
import math
import numpy as np
from .base_layer_weight import BaseLayerWeight


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
        # llama and baichuan has same struct, but weight name is not same.
        if self.network_config.get("model_type", "llama") == "baichuan":
            # q k v weights for baichuan
            self.load_baichuan_qkvo_weight(weights)
        else:
            # q k v weights for llama
            self.load_llama_qkvo_weight(weights)
            
        # ffn params
        self.load_ffn_weights(weights)
        return
    
    def load_llama_qkvo_weight(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.input_layernorm = weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"].to(self.data_type_).cuda()

        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self.q_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.q_weight_ = self.q_weight_.cuda()
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            self.k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self.k_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.k_weight_ = self.k_weight_.cuda()
        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            self.v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self.v_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.v_weight_ = self.v_weight_.cuda()
        
        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.att_out_dense_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.att_out_dense_weight_ = self.att_out_dense_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.att_out_dense_weight_ = self.att_out_dense_weight_.cuda()
        return
    
    def load_baichuan_qkvo_weight(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.input_layernorm = weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"].to(self.data_type_).cuda()

        n_embed = self.network_config["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for baichuan
        if f"model.layers.{self.layer_num_}.self_attn.W_pack.weight" in weights:
            qkv_weights = weights[f"model.layers.{self.layer_num_}.self_attn.W_pack.weight"]
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
        
        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.att_out_dense_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.att_out_dense_weight_ = self.att_out_dense_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.att_out_dense_weight_ = self.att_out_dense_weight_.cuda()
        return
    
    def load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.post_attention_layernorm_weight_ = weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"].to(
                self.data_type_).cuda()
    
        inter_size = self.network_config['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            self.up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = self.up_proj.transpose(0, 1).contiguous().to(self.data_type_)
            self.up_proj = self.up_proj.cuda()

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            self.gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = self.gate_proj.transpose(0, 1).contiguous().to(self.data_type_)

            self.gate_proj = self.gate_proj.cuda()


        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self.down_proj.transpose(0, 1).contiguous().to(self.data_type_)
            
            self.down_proj = self.down_proj.cuda()
        return