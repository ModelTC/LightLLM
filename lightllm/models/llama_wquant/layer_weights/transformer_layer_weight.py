import math

import numpy as np
import torch
from functools import partial

from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import quantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import quantize_int4
from lightllm.common.basemodel.cuda_kernel.lmdeploy_wquant import quantize_int4_lmdeploy


class LlamaTransformerLayerWeightQuantized(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.init_quant_mode()
    
    def init_quant_mode(self):
        if "triton_int8weight" in self.mode:
            self.quantize_weight = partial(quantize_int8, tp_rank=self.tp_rank_)
        if "triton_int4weight" in self.mode:
            self.int4_q_group_size = 128
            for _mode in self.mode:
                if _mode.startswith('g'):
                    self.int4_q_group_size = int(_mode[1:])
            self.quantize_weight = partial(quantize_int4, group_size=self.int4_q_group_size, tp_rank=self.tp_rank_)
        if "lmdeploy_int4weight" in self.mode:
            self.int4_q_group_size = 128
            for _mode in self.mode:
                if _mode.startswith('g'):
                    self.int4_q_group_size = int(_mode[1:])
            self.quantize_weight = partial(quantize_int4_lmdeploy, group_size=self.int4_q_group_size, tp_rank=self.tp_rank_)
        

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.qkv_weight_,
            self.o_weight_,
            self.ffn_norm_weight_,
            self.gate_up_proj,
            self.down_proj
            ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = n_embed // self.network_config_["num_attention_heads"] * self.network_config_["num_key_value_heads"] // self.world_size_
        
        if getattr(self, "qkv_weight_", None) is None:
            self.qkv_weight_ = torch.empty(n_embed, q_split_n_embed + 2 * kv_split_n_embed, dtype=self.data_type_, device='cpu')
            self.qkv_step_ = 0
        
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"]
            q_weight_ = q_weight_[q_split_n_embed * self.tp_rank_: q_split_n_embed * (self.tp_rank_ + 1), :]
            q_weight_ = q_weight_.transpose(0, 1).to(self.data_type_)
            self.qkv_weight_[:, :q_split_n_embed] = q_weight_
            self.qkv_step_ += 1

        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"]
            k_weight_ = k_weight_[kv_split_n_embed *  self.tp_rank_: kv_split_n_embed * (self.tp_rank_ + 1), :]
            k_weight_ = k_weight_.transpose(0, 1).to(self.data_type_)
            self.qkv_weight_[:, q_split_n_embed: (q_split_n_embed + kv_split_n_embed)] = k_weight_
            self.qkv_step_ += 1

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"]
            v_weight_ = v_weight_[kv_split_n_embed * self.tp_rank_: kv_split_n_embed * (self.tp_rank_ + 1), :]
            v_weight_ = v_weight_.transpose(0, 1).to(self.data_type_)
            self.qkv_weight_[:, (q_split_n_embed + kv_split_n_embed):(q_split_n_embed + 2 * kv_split_n_embed)] = v_weight_
            self.qkv_step_ += 1
        
        if self.qkv_step_ == 3:
            self.qkv_step_ = 0
            self.qkv_weight_ = self.quantize_weight(self.qkv_weight_)

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            self.o_weight_ = self.o_weight_[:, q_split_n_embed * self.tp_rank_: q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self.quantize_weight(self.o_weight_.transpose(0, 1))
        
        return
    
    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])
    
        n_embed = self.network_config_["hidden_size"]
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if getattr(self, "gate_up_proj", None) is None:
            self.gate_up_proj = torch.empty(n_embed, split_inter_size * 2, dtype=self.data_type_, device='cpu')
            self.gate_up_step = 0

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                        self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            gate_proj = gate_proj.transpose(0, 1).to(self.data_type_)
            self.gate_up_proj[:, : split_inter_size] = gate_proj
            self.gate_up_step += 1

        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                    self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            up_proj = up_proj.transpose(0, 1).to(self.data_type_)
            self.gate_up_proj[:, split_inter_size : split_inter_size * 2] = up_proj
            self.gate_up_step += 1

        if self.gate_up_step == 2:
            self.gate_up_step = 0
            self.gate_up_proj = self.quantize_weight(self.gate_up_proj)

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self.quantize_weight(self.down_proj.transpose(0, 1))

        return
