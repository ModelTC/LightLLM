import torch
import math
import numpy as np
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight


class StarcoderTransformerLayerWeight(BloomTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        assert network_config["num_attention_heads"] % self.world_size_ == 0

    def init_static_params(self):
        pass

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"transformer.h.{self.layer_num_}.ln_1.weight" in weights:
            self.att_norm_weight_ = weights[f"transformer.h.{self.layer_num_}.ln_1.weight"].to(self.data_type_).cuda()

        if f"transformer.h.{self.layer_num_}.ln_1.bias" in weights:
            self.att_norm_bias_ = weights[f"transformer.h.{self.layer_num_}.ln_1.bias"].to(self.data_type_).cuda()

        # attention params
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        split_n_embed = n_embed // self.world_size_
        if f"transformer.h.{self.layer_num_}.attn.c_attn.weight" in weights:
            qkv_weight_ = (
                weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"]
                .transpose(0, 1)
                .contiguous()
                .to(self.data_type_)
            )
            self.q_weight_ = qkv_weight_[:, :n_embed][
                :, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
            ]
            self.q_weight_ = self.q_weight_.cuda()

            self.k_weight_ = qkv_weight_[:, n_embed : n_embed + head_dim]
            self.v_weight_ = qkv_weight_[:, n_embed + head_dim : n_embed + 2 * head_dim]

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        if f"transformer.h.{self.layer_num_}.attn.c_attn.bias" in weights:

            qkv_bias_ = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"].to(self.data_type_)
            self.q_bias_ = self._cuda(
                qkv_bias_[:n_embed][split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            )
            self.k_bias_ = qkv_bias_[n_embed : n_embed + head_dim]
            self.v_bias_ = qkv_bias_[n_embed + head_dim : n_embed + 2 * head_dim]

        self._try_cat_to(["k_bias_", "v_bias_"], "kv_bias_", cat_dim=0)

        # attention output dense params
        if f"transformer.h.{self.layer_num_}.attn.c_proj.weight" in weights:
            self.o_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.weight"][
                :, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
            ]
            self.o_weight_ = self.o_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.o_weight_ = self.o_weight_.cuda()

        if f"transformer.h.{self.layer_num_}.attn.c_proj.bias" in weights:
            self.o_bias_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.bias"].to(self.data_type_)
            self.o_bias_ = self.o_bias_.cuda()

    def _load_ffn_weights(self, weights):
        if f"transformer.h.{self.layer_num_}.ln_2.weight" in weights:
            self.ffn_norm_weight_ = weights[f"transformer.h.{self.layer_num_}.ln_2.weight"].to(self.data_type_).cuda()
        if f"transformer.h.{self.layer_num_}.ln_2.bias" in weights:
            self.ffn_norm_bias_ = weights[f"transformer.h.{self.layer_num_}.ln_2.bias"].to(self.data_type_).cuda()

        # ffn params
        n_embed = self.network_config_["hidden_size"]
        intermediate_size = n_embed * 4
        split_inter_size = intermediate_size // self.world_size_
        if f"transformer.h.{self.layer_num_}.mlp.c_fc.weight" in weights:
            self.ffn_1_weight_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_fc.weight"].to(self.data_type_)
            self.ffn_1_weight_ = (
                self.ffn_1_weight_[split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
                .transpose(0, 1)
                .contiguous()
                .cuda()
            )

        if f"transformer.h.{self.layer_num_}.mlp.c_fc.bias" in weights:
            self.ffn_1_bias_ = (
                weights[f"transformer.h.{self.layer_num_}.mlp.c_fc.bias"][
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
                ]
                .to(self.data_type_)
                .contiguous()
                .cuda()
            )

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.weight" in weights:
            self.ffn_2_weight_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"].to(self.data_type_)
            self.ffn_2_weight_ = (
                self.ffn_2_weight_[:, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)]
                .transpose(0, 1)
                .contiguous()
                .cuda()
            )

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.bias" in weights:
            self.ffn_2_bias_ = (
                weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.bias"].to(self.data_type_).contiguous().cuda()
            )

        return
