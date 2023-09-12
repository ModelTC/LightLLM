import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class ChatGLM2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        assert network_config["num_attention_heads"] % self.world_size_ == 0

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
                   self.ffn_1_weight_,
                   self.ffn_2_weight_,
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"transformer.encoder.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.input_layernorm.weight"].to(self.data_type_).cuda()

        # attention params
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        multi_query_group_num = self.network_config_["multi_query_group_num"]
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
            self.o_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.self_attention.dense.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self.o_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.o_weight_ = self.o_weight_.cuda()

    def _load_ffn_weights(self, weights):
        if f"transformer.encoder.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.post_attention_layernorm.weight"].to(
                self.data_type_).cuda()

        # ffn params
        intermediate_size =self.network_config_['ffn_hidden_size'] * 2
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
