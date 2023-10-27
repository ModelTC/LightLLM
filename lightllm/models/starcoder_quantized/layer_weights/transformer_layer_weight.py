import torch
import math
import numpy as np
from functools import partial

from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import quantize_int8
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import quantize_int4


class StarcoderTransformerLayerWeightQuantized(BloomTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], group_size=128):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        assert network_config["num_attention_heads"] % self.world_size_ == 0
        quantize_func_dict = {
            'int8weight': quantize_int8,
            'int4weight': partial(quantize_int4, group_size=group_size)
        }
        self.quantize_weight = None
        for item in mode:
            if item in quantize_func_dict:
                self.quantize_weight = quantize_func_dict[item]

    def init_static_params(self):
        pass

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.att_norm_bias_,
                   self.qkv_fused_weight,
                   self.qkv_fused_bias,
                   self.o_weight_,
                   self.o_bias_,

                   self.ffn_norm_weight_,
                   self.ffn_norm_bias_,
                   self.ffn_1_weight_,
                   self.ffn_1_bias_,
                   self.ffn_2_weight_,
                   self.ffn_2_bias_,
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

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
            self.qkv_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"].transpose(0, 1).contiguous().to(self.data_type_)
            self.q_weight_ = self.qkv_weight_[:, :n_embed][:, split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]

            self.k_weight_ = self.qkv_weight_[:, n_embed:n_embed + head_dim]

            self.v_weight_ = self.qkv_weight_[:, n_embed + head_dim:n_embed + 2 * head_dim]

            self.qkv_fused_weight = torch.cat((self.q_weight_, self.k_weight_, self.v_weight_), dim=1)
            self.qkv_fused_weight, self.qkv_fused_weight_scale, self.qkv_fused_weight_zp = \
                self.quantize_weight(self.qkv_fused_weight)
            self.qkv_fused_weight = self.qkv_fused_weight.cuda()
            self.qkv_fused_weight_scale = self.qkv_fused_weight_scale.to(self.data_type_).cuda()
            self.qkv_fused_weight_zp = self.qkv_fused_weight_zp.cuda() \
                if self.qkv_fused_weight_zp is not None else None

        if f"transformer.h.{self.layer_num_}.attn.c_attn.bias" in weights:
            self.qkv_bias_ = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"].to(self.data_type_)
            self.q_bias_ = self.qkv_bias_[:n_embed][split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.k_bias_ = self.qkv_bias_[n_embed : n_embed + head_dim]
            self.v_bias_ = self.qkv_bias_[n_embed + head_dim : n_embed + 2 * head_dim]
 
            self.qkv_fused_bias = torch.cat((self.q_bias_, self.k_bias_, self.v_bias_), dim=0).cuda()

        # attention output dense params
        if f"transformer.h.{self.layer_num_}.attn.c_proj.weight" in weights:
            self.o_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self.o_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.o_weight_, self.o_weight_scale_, self.o_weight_zp_ = \
                self.quantize_weight(self.o_weight_)
            self.o_weight_ = self.o_weight_.cuda()
            self.o_weight_scale_ = self.o_weight_scale_.to(self.data_type_).cuda()
            self.o_weight_zp_ = self.o_weight_zp_.cuda() if self.o_weight_zp_ is not None else None

        if f"transformer.h.{self.layer_num_}.attn.c_proj.bias" in weights:
            self.o_bias_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.bias"].to(self.data_type_)
            self.o_bias_ = self.o_bias_.cuda()

    def _load_ffn_weights(self, weights):
        if f"transformer.h.{self.layer_num_}.ln_2.weight" in weights:
            self.ffn_norm_weight_ = weights[f"transformer.h.{self.layer_num_}.ln_2.weight"].to(
                self.data_type_).cuda()
        if f"transformer.h.{self.layer_num_}.ln_2.bias" in weights:
            self.ffn_norm_bias_ = weights[f"transformer.h.{self.layer_num_}.ln_2.bias"].to(
                self.data_type_).cuda()

        # ffn params
        n_embed = self.network_config_["hidden_size"]
        intermediate_size = n_embed * 4
        split_inter_size = intermediate_size // self.world_size_
        if f"transformer.h.{self.layer_num_}.mlp.c_fc.weight" in weights:
            self.ffn_1_weight_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_fc.weight"].to(self.data_type_)
            self.ffn_1_weight_ = self.ffn_1_weight_[split_inter_size * self.tp_rank_: split_inter_size *
                                                    (self.tp_rank_ + 1), :].transpose(0, 1).contiguous().to(self.data_type_)

            self.ffn_1_weight_, self.ffn_1_weight_scale, self.ffn_1_weight_zp = \
                self.quantize_weight(self.ffn_1_weight_)
            self.ffn_1_weight_ = self.ffn_1_weight_.cuda()
            self.ffn_1_weight_scale = self.ffn_1_weight_scale.to(self.data_type_).cuda()
            self.ffn_1_weight_zp = self.ffn_1_weight_zp.cuda() if self.ffn_1_weight_zp is not None else None

        if f"transformer.h.{self.layer_num_}.mlp.c_fc.bias" in weights:
            self.ffn_1_bias_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_fc.bias"][split_inter_size *
                                                                                      self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)].to(self.data_type_).contiguous().cuda()

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.weight" in weights:
            self.ffn_2_weight_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"].to(self.data_type_)
            self.ffn_2_weight_ = self.ffn_2_weight_[:, split_inter_size *
                                                    self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)].transpose(0, 1).contiguous().to(self.data_type_)
            self.ffn_2_weight_, self.ffn_2_weight_scale, self.ffn_2_weight_zp = \
                self.quantize_weight(self.ffn_2_weight_)
            self.ffn_2_weight_ = self.ffn_2_weight_.cuda()
            self.ffn_2_weight_scale = self.ffn_2_weight_scale.to(self.data_type_).cuda()
            self.ffn_2_weight_zp = self.ffn_2_weight_zp.cuda() if self.ffn_2_weight_zp is not None else None

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.bias" in weights:
            self.ffn_2_bias_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.bias"].to(self.data_type_).contiguous().cuda()

        return
