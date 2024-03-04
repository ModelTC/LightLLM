import math

import numpy as np
import torch
from functools import partial

from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.cuda_kernel.ppl_awquant import dynamic_channelwise_quant_fp16_i8_ppl


class LlamaTransformerLayerActivationWeightQuantized(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.init_quant_mode()

    def init_quant_mode(self):
        if "ppl_w8a8" in self.mode:
            self.quantize_weight = partial(dynamic_channelwise_quant_fp16_i8_ppl, tp_rank=self.tp_rank_)
        else:
            raise Exception(f"error mode {self.mode}")

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_weight_,
            self.o_weight_,
            self.ffn_norm_weight_,
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = (
            n_embed
            // self.network_config_["num_attention_heads"]
            * self.network_config_["num_key_value_heads"]
            // self.world_size_
        )

        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"]
            q_weight_ = q_weight_[q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1), :]
            q_weight_ = q_weight_.transpose(0, 1).to(self.data_type_)
            self.q_weight_ = self.quantize_weight(q_weight_)

        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"]
            k_weight_ = k_weight_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = k_weight_.transpose(0, 1).to(self.data_type_)

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"]
            v_weight_ = v_weight_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = v_weight_.transpose(0, 1).to(self.data_type_)

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1, handle_func=self.quantize_weight)

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            o_weight_ = o_weight_[:, q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(o_weight_.transpose(0, 1))
        return

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(
                weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"]
            )

        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.up_proj = self.quantize_weight(up_proj.transpose(0, 1).to(self.data_type_))

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.gate_proj = self.quantize_weight(gate_proj.transpose(0, 1).to(self.data_type_))

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.down_proj = self.quantize_weight(down_proj.transpose(0, 1))
        return
