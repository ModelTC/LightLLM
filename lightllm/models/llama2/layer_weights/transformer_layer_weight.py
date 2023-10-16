import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight

class Llama2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return
    
    def _load_qkvo_weights(self, weights):
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        # attention params
        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        split_key_value_embed = n_embed // self.network_config_["num_attention_heads"] * self.network_config_["num_key_value_heads"] // self.world_size_
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            self.k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_key_value_embed *
                                                                                                self.tp_rank_: split_key_value_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(self.k_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            self.v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_key_value_embed *
                                                                                                self.tp_rank_: split_key_value_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(self.v_weight_.transpose(0, 1))

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))
        return

