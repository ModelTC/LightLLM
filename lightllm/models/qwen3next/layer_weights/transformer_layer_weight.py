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
from lightllm.models.qwen3next.layer_weights.gdn_layer_weight import Qwen3NextGatedDeltaNetWeight


class Qwen3NextTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    @override
    def _parse_config(self):
        super()._parse_config()
        self.full_attention_interval = self.network_config_["full_attention_interval"]
        self.is_gdn = (self.layer_num_ + 1) % self.full_attention_interval != 0
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

        if self.is_gdn:
            self.gdn_layer_weight = Qwen3NextGatedDeltaNetWeight(
                self.layer_num_, self.data_type_, self.network_config_, self.mode, self.quant_cfg
            )
        else:
            self._init_qkv()
            self._init_o()
            self.q_norm_weight_ = NormWeight(weight_name=self._q_norm_name, data_type=self.data_type_)
            self.k_norm_weight_ = NormWeight(weight_name=self._k_norm_name, data_type=self.data_type_)
            self.o_gate_proj = ROWMMWeight(
                weight_name=f"model.layers.{self.layer_num_}.self_attn.o_gate_proj.weight",
                data_type=self.data_type_,
                bias_name=f"model.layers.{self.layer_num_}.self_attn.o_gate_proj.bias",
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="o_gate_proj",
            )
        return

    @override
    def load_hf_weights(self, weights):
        if not self.is_gdn:
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
