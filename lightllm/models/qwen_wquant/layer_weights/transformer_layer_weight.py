import torch
import math
import numpy as np
from functools import partial
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.models.llama_wquant.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeightQuantized


class QwenTransformerLayerWeightQuantized(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        LlamaTransformerLayerWeightQuantized.init_quant_mode(self)
        return

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

            self.q_weight_ = q_weights[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ].transpose(0, 1)
            self.k_weight_ = k_weights[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ].transpose(0, 1)
            self.v_weight_ = v_weights[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ].transpose(0, 1)

            self.q_weight_ = self.quantize_weight(self.q_weight_)

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1, handle_func=self.quantize_weight)

        if f"transformer.h.{self.layer_num_}.attn.c_attn.bias" in weights:
            qkv_bias = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"]
            split_size = qkv_bias.shape[0] // 3
            q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size, dim=0)
            self.q_bias_ = self._cuda(q_bias[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)])
            self.k_bias_ = k_bias[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            self.v_bias_ = v_bias[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]

        self._try_cat_to(["k_bias_", "v_bias_"], "kv_bias_", cat_dim=0)

        # attention output dense params
        if f"transformer.h.{self.layer_num_}.attn.c_proj.weight" in weights:
            self.o_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_proj.weight"][
                :, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
            ]
            self.o_weight_ = self.quantize_weight(self.o_weight_.transpose(0, 1))

        if f"transformer.h.{self.layer_num_}.ln_2.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"transformer.h.{self.layer_num_}.ln_2.weight"])
        # ffn params
        inter_size = self.network_config_["intermediate_size"] // 2
        split_inter_size = inter_size // self.world_size_

        if f"transformer.h.{self.layer_num_}.mlp.w2.weight" in weights:
            gate_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w2.weight"]
            gate_proj = gate_proj[split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = gate_proj.transpose(0, 1).to(self.data_type_)

        if f"transformer.h.{self.layer_num_}.mlp.w1.weight" in weights:
            up_proj = weights[f"transformer.h.{self.layer_num_}.mlp.w1.weight"]
            up_proj = up_proj[split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = up_proj.transpose(0, 1).to(self.data_type_)

        self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=1, handle_func=self.quantize_weight)

        if f"transformer.h.{self.layer_num_}.mlp.c_proj.weight" in weights:
            self.down_proj = weights[f"transformer.h.{self.layer_num_}.mlp.c_proj.weight"]
            self.down_proj = self.down_proj[
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.down_proj = self.quantize_weight(self.down_proj.transpose(0, 1))

        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_weight_,
            self.q_bias_,
            self.kv_bias_,
            self.o_weight_,
            self.ffn_norm_weight_,
            self.gate_up_proj,
            self.down_proj,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
