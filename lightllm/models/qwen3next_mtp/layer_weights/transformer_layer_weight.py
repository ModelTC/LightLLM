import os
import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.utils.envs_utils import enable_env_vars
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    NormWeight,
)
from functools import partial
from typing_extensions import override


class Qwen3NextMTPTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    @override
    def _init_weight_names(self):
        self._q_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_norm_name = f"mtp.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._q_bias_name = None
        self._k_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_norm_name = f"mtp.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._k_bias_name = None
        self._v_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._kv_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.kv_proj.weight"
        self._kv_bias_name = None
        self._o_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._att_norm_weight_name = f"mtp.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"mtp.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

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

        self._init_qkv()
        self._init_o()
        self.q_norm_weight_ = NormWeight(weight_name=self._q_norm_name, data_type=self.data_type_)
        self.k_norm_weight_ = NormWeight(weight_name=self._k_norm_name, data_type=self.data_type_)
        self._o_gate_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.o_gate_proj.weight"
        self.o_gate_proj = ROWMMWeight(
            weight_names=self._o_gate_weight_name,
            data_type=self.data_type_,
            bias_names=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="o_gate_proj",
        )
        return

    @override
    def load_hf_weights(self, weights):
        self._split_q_with_gate(weights)
        super().load_hf_weights(weights)

    def _init_shared_expert_weight(self):
        prefix = f"mtp.layers.{self.layer_num_}.mlp.shared_expert"
        self.shared_expert_gate_up_proj = ROWMMWeight(
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate_up_proj",
        )
        self.shared_expert_down_proj = COLMMWeight(
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_down_proj",
        )
        self.shared_expert_gate = ROWMMWeight(
            weight_names=f"mtp.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate",
            tp_rank=0,
            tp_world_size=1,
        )

    def _split_q_with_gate(self, weights):
        if self._q_weight_name in weights:
            weight = weights[self._q_weight_name]
            num_heads = self.tp_q_head_num_ * self.tp_world_size_
            weight = weight.view(num_heads * 2, self.head_dim, -1)
            _q_proj = weight[0::2].reshape(-1, weight.shape[-1])
            _gate_proj = weight[1::2].reshape(-1, weight.shape[-1])
            weights[self._q_weight_name] = _q_proj
            weights[self._o_gate_weight_name] = _gate_proj
