import torch
import math
import numpy as np
from functools import partial

from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import quantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import quantize_int4
from lightllm.common.basemodel import TransformerLayerWeight

class StarcoderTransformerLayerWeightQuantized(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        assert network_config["num_attention_heads"] % self.world_size_ == 0
        self.init_quant_mode()
    
    def init_quant_mode(self):
        if "int8weight" in self.mode:
            self.quantize_weight = partial(quantize_int8, tp_rank=self.tp_rank_)
        if "int4weight" in self.mode:
            self.int4_q_group_size = 128
            for _mode in self.mode:
                if _mode.startswith('g'):
                    self.int4_q_group_size = int(_mode[1:])
            self.quantize_weight = partial(quantize_int4, group_size=self.int4_q_group_size, tp_rank=self.tp_rank_)
        return
    
    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.att_norm_bias_,
                   self.qkv_weight_,
                   self.qkv_bias_,
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
            self.att_norm_weight_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_1.weight"])

        if f"transformer.h.{self.layer_num_}.ln_1.bias" in weights:
            self.att_norm_bias_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_1.bias"])

        # attention params
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        split_n_embed = n_embed // self.world_size_
        if f"transformer.h.{self.layer_num_}.attn.c_attn.weight" in weights:
            qkv_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"].transpose(0, 1).contiguous().to(self.data_type_)
            q_weight_ = qkv_weight_[:, :n_embed][:, split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            k_weight_ = qkv_weight_[:, n_embed:n_embed + head_dim]
            v_weight_ = qkv_weight_[:, n_embed + head_dim:n_embed + 2 * head_dim]
            qkv_fused_weight = torch.cat((q_weight_, k_weight_, v_weight_), dim=1)

            self.qkv_weight_ = self.quantize_weight(qkv_fused_weight)

        if f"transformer.h.{self.layer_num_}.attn.c_attn.bias" in weights:
            qkv_bias_ = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"].to(self.data_type_)
            q_bias_ = qkv_bias_[:n_embed][split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            k_bias_ = qkv_bias_[n_embed : n_embed + head_dim]
            v_bias_ = qkv_bias_[n_embed + head_dim : n_embed + 2 * head_dim]
 
            self.qkv_bias_ = self._cuda(torch.cat((q_bias_, k_bias_, v_bias_), dim=0))

        # attention output dense params
        if f"transformer.h.{self.layer_num_}.attn.c_proj.weight" in weights:
            o_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.weight"]
            o_weight_ = o_weight_[:,split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            o_weight_ = o_weight_.transpose(0, 1).contiguous().to(self.data_type_)
            self.o_weight_ = self.quantize_weight(o_weight_)

        if f"transformer.h.{self.layer_num_}.attn.c_proj.bias" in weights:
            o_bias_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.bias"].to(self.data_type_)
            o_bias_ = o_bias_ / self.world_size_
            self.o_bias_ = self._cuda(o_bias_)

    def _load_ffn_weights(self, weights):
        if f"transformer.h.{self.layer_num_}.ln_2.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_2.weight"])
        if f"transformer.h.{self.layer_num_}.ln_2.bias" in weights:
            self.ffn_norm_bias_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_2.bias"])

        # ffn params
        n_embed = self.network_config_["hidden_size"]
        intermediate_size = n_embed * 4
        split_inter_size = intermediate_size // self.world_size_
        if f"transformer.h.{self.layer_num_}.mlp.c_fc.weight" in weights:
            self.ffn_1_weight_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_fc.weight"].to(self.data_type_)
            self.ffn_1_weight_ = self.ffn_1_weight_[split_inter_size * self.tp_rank_: split_inter_size *
                                                    (self.tp_rank_ + 1), :].transpose(0, 1).contiguous().to(self.data_type_)
            self.ffn_1_weight_ = self.quantize_weight(self.ffn_1_weight_)

        if f"transformer.h.{self.layer_num_}.mlp.c_fc.bias" in weights:
            self.ffn_1_bias_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_fc.bias"]
            self.ffn_1_bias_ = self.ffn_1_bias_[split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.ffn_1_bias_ = self._cuda(self.ffn_1_bias_)

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.weight" in weights:
            self.ffn_2_weight_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"].to(self.data_type_)
            self.ffn_2_weight_ = self.ffn_2_weight_[:, split_inter_size *
                                                    self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)].transpose(0, 1).contiguous().to(self.data_type_)
            self.ffn_2_weight_ = self.quantize_weight(self.ffn_2_weight_)

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.bias" in weights:
            self.ffn_2_bias_ = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.bias"]
            self.ffn_2_bias_ = self.ffn_2_bias_ / self.world_size_
            self.ffn_2_bias_ = self._cuda(self.ffn_2_bias_)

        return
