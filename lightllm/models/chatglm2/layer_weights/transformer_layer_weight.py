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
        self.mode = mode
        self.network_config = network_config
        assert network_config["num_attention_heads"] % self.world_size_ == 0

    def load_hf_weights(self, weights):
        # input layernorm params
        if f"transformer.encoder.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.input_layernorm_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.input_layernorm.weight"].to(self.data_type_).cuda()

        # attention params
        n_embed = self.network_config["hidden_size"]
        head_dim = self.network_config["hidden_size"] // self.network_config["num_attention_heads"]
        multi_query_group_num = self.network_config["multi_query_group_num"]
        split_n_embed = n_embed // self.world_size_
        if f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.weight" in weights:
            self.qkv_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.weight"].transpose(0, 1).contiguous().to(self.data_type_)
            self.q_weight_ = self.qkv_weight_[:, :n_embed][:, split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.q_weight_ = self.q_weight_.cuda()
            self.k_weight_ = self.qkv_weight_[:, n_embed:n_embed + head_dim * multi_query_group_num]
            self.k_weight_ = self.k_weight_.cuda()

            self.v_weight_ = self.qkv_weight_[:, n_embed + multi_query_group_num * head_dim : n_embed + 2 * multi_query_group_num * head_dim]
            self.v_weight_ = self.v_weight_.cuda()

        if f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.bias" in weights:

            self.qkv_bias_ = weights[f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.bias"].to(self.data_type_)
            self.q_bias_ = self.qkv_bias_[:n_embed].cuda()[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.k_bias_ = self.qkv_bias_[n_embed : n_embed + head_dim * multi_query_group_num].cuda()
            self.v_bias_ = self.qkv_bias_[n_embed + multi_query_group_num * head_dim : n_embed + 2 * multi_query_group_num * head_dim].cuda()

        # attention output dense params
        if f"transformer.encoder.layers.{self.layer_num_}.self_attention.dense.weight" in weights:
            self.att_out_dense_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.self_attention.dense.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.att_out_dense_weight_ = self.att_out_dense_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.att_out_dense_weight_ = self.att_out_dense_weight_.cuda()

        if f"transformer.encoder.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.post_attention_layernorm_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.post_attention_layernorm.weight"].to(
                self.data_type_).cuda()

        # ffn params
        intermediate_size =self.network_config['ffn_hidden_size'] * 2
        split_inter_size = intermediate_size // self.world_size_
        if f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_h_to_4h.weight" in weights:
            self.ffn_1_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_h_to_4h.weight"].to(self.data_type_)
            self.ffn_1_weight_ = self.ffn_1_weight_[split_inter_size * self.tp_rank_: split_inter_size *
                                                    (self.tp_rank_ + 1), :].transpose(0, 1).contiguous().cuda()
        split_inter_size = split_inter_size // 2    
        if f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_4h_to_h.weight" in weights:
            self.ffn_2_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_4h_to_h.weight"].to(self.data_type_)
            self.ffn_2_weight_ = self.ffn_2_weight_[:, split_inter_size *
                                                    self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)].transpose(0, 1).contiguous().cuda()

        return
