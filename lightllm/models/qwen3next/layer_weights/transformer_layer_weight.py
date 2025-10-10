import os
import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    MultiROWMMWeight,
    COLMMWeight,
    NormWeight,
    FusedMoeWeightTP,
    FusedMoeWeightEP,
    ROWBMMWeight,
)
from functools import partial
from typing_extensions import override
from lightllm.common.basemodel.layer_weights.meta_weights import TpParameterWeight


class Qwen3NextTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    @override
    def _parse_config(self):
        super()._parse_config()
        self.full_attention_interval = self.network_config_["full_attention_interval"]
        self.is_linear = (self.layer_num_ + 1) % self.full_attention_interval != 0
        if self.is_linear:
            self._parse_linear_config()
        return

    @override
    def _init_weight(self):
        self._init_moe()
        self._init_shared_expert_weight()

        self.att_norm_weight_ = NormWeight(
            self._att_norm_weight_name, self.data_type_, bias_name=self._att_norm_bias_name
        )
        self.ffn_norm_weight_ = NormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )

        if self.is_linear:
            self._init_linear_weight()
        else:
            self._init_qkv()
            self._init_o()
            self.q_norm_weight_ = NormWeight(weight_name=self._q_norm_name, data_type=self.data_type_)
            self.k_norm_weight_ = NormWeight(weight_name=self._k_norm_name, data_type=self.data_type_)
            self._o_gate_weight_name = f"model.layers.{self.layer_num_}.self_attn.o_gate_proj.weight"
            self._o_gate_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_gate_proj.bias"
            self.o_gate_proj = ROWMMWeight(
                weight_name=self._o_gate_weight_name,
                data_type=self.data_type_,
                bias_name=self._o_gate_bias_name,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="o_gate_proj",
            )
            self._gate = None
        return

    @override
    def load_hf_weights(self, weights):
        if self.is_linear:
            if self.linear_conv1d.weight_name in weights:
                weights[self.linear_conv1d.weight_name] = self._parse_linear_conv1d(
                    weights[self.linear_conv1d.weight_name].squeeze(1)
                )
            if self.linear_conv1d.bias_name in weights:
                weights[self.linear_conv1d.bias_name] = self._parse_linear_conv1d(weights[self.linear_conv1d.bias_name])
        else:
            self._split_q_with_gate(weights)
        super().load_hf_weights(weights)

    def _init_shared_expert_weight(self):
        prefix = f"model.layers.{self.layer_num_}.mlp.shared_expert"
        self.shared_expert_gate_up_proj = MultiROWMMWeight(
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate_up_proj",
        )
        self.shared_expert_down_proj = COLMMWeight(
            weight_name=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_down_proj",
        )
        self.shared_expert_gate = ROWMMWeight(
            weight_name=f"model.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_name=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate",
            tp_rank=0,
            tp_world_size=1,
        )

    def _split_q_with_gate(self, weights):
        if self.q_proj.weight_name in weights:
            weight = weights[self.q_proj.weight_name]
            num_heads = self.tp_q_head_num_ * self.tp_world_size_
            weight = weight.view(num_heads * 2, self.head_dim, -1)
            _q_proj = weight[0::2].reshape(-1, weight.shape[-1])
            _gate_proj = weight[1::2].reshape(-1, weight.shape[-1])
            weights[self.q_proj.weight_name] = _q_proj
            weights[self.o_gate_proj.weight_name] = _gate_proj
        if self.q_proj.bias_name in weights:
            bias = weights[self.q_proj.bias_name]
            num_heads = self.tp_q_head_num_ * self.tp_world_size_
            bias = bias.view(num_heads * 2, self.head_dim)
            _q_proj = bias[0::2].reshape(-1)
            _gate_proj = bias[1::2].reshape(-1)
            weights[self.q_proj.bias_name] = _q_proj
            weights[self.o_gate_proj.bias_name] = _gate_proj

    def _parse_linear_conv1d(self, weight):
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim

        q_bias, k_bias, v_bias = torch.split(weight, [qk_dim, qk_dim, v_dim], dim=0)
        q_splits = q_bias.chunk(self.tp_world_size_, dim=0)
        k_splits = k_bias.chunk(self.tp_world_size_, dim=0)
        v_splits = v_bias.chunk(self.tp_world_size_, dim=0)

        new_weight = torch.cat(
            [torch.cat([q_splits[i], k_splits[i], v_splits[i]], dim=0) for i in range(self.tp_world_size_)], dim=0
        )

        return new_weight

    def _parse_linear_config(self):
        self.linear_num_v_heads = self.network_config_["linear_num_value_heads"]
        self.linear_num_k_heads = self.network_config_["linear_num_key_heads"]
        self.linear_k_head_dim = self.network_config_["linear_key_head_dim"]
        self.linear_v_head_dim = self.network_config_["linear_value_head_dim"]

    def _init_linear_weight(self):
        prefix = f"model.layers.{self.layer_num_}.linear_attn"
        self.linear_conv1d = ROWMMWeight(
            weight_name=f"{prefix}.conv1d.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="conv1d_weight",
        )

        self.linear_in_proj = MultiROWMMWeight(
            weight_names=[f"{prefix}.in_proj_qkvz.weight", f"{prefix}.in_proj_ba.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="in_proj_weight",
        )

        self.linear_out_proj = COLMMWeight(
            weight_name=f"{prefix}.out_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="out_proj_weight",
        )

        self.linear_dt_bias = TpParameterWeight(
            weight_name=f"{prefix}.dt_bias",
            data_type=torch.float32,
            split_n_embed=self.linear_num_v_heads // self.tp_world_size_,
        )

        self.linear_A_log = TpParameterWeight(
            weight_name=f"{prefix}.A_log",
            data_type=torch.float32,
            split_n_embed=self.linear_num_v_heads // self.tp_world_size_,
        )

        self.linear_norm = NormWeight(
            weight_name=f"{prefix}.norm.weight",
            data_type=self.data_type_,
        )
