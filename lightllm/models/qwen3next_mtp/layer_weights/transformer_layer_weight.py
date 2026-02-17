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
    RMSNormWeight,
    QKRMSNORMWeight,
    KVROWNMMWeight,
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
    def _init_qkv(self):
        # Override parent's QKVROWNMMWeight which requires kv_head_num % tp == 0.
        # Qwen3-Next has few KV heads; KVROWNMMWeight handles repeating.
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        self.q_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_method=self.get_quant_method("q_proj"),
        )
        self.kv_proj = KVROWNMMWeight(
            in_dim=in_dim,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("kv_proj"),
        )

    @override
    def _init_weight(self):
        self._init_moe()
        self._init_shared_expert_weight()

        hidden_size = self.network_config_["hidden_size"]
        self.att_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._att_norm_weight_name,
            data_type=self.data_type_,
        )
        self.ffn_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._ffn_norm_weight_name,
            data_type=self.data_type_,
        )

        self._init_qkv()
        self._init_o()
        self.q_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim, weight_name=self._q_norm_name, data_type=self.data_type_
        )
        self.k_norm_weight_ = QKRMSNORMWeight(
            dim=self.head_dim, weight_name=self._k_norm_name, data_type=self.data_type_
        )
        self._o_gate_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.o_gate_proj.weight"
        q_out_dim = self.q_head_num_ * self.head_dim
        self.o_gate_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[q_out_dim],
            weight_names=self._o_gate_weight_name,
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("o_gate_proj"),
        )
        return

    @override
    def load_hf_weights(self, weights):
        self._split_q_with_gate(weights)
        super().load_hf_weights(weights)

    def _init_shared_expert_weight(self):
        prefix = f"mtp.layers.{self.layer_num_}.mlp.shared_expert"
        hidden_size = self.network_config_["hidden_size"]
        shared_inter = self.network_config_["shared_expert_intermediate_size"]
        self.shared_expert_gate_up_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[shared_inter, shared_inter],
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_expert_gate_up_proj"),
        )
        self.shared_expert_down_proj = COLMMWeight(
            in_dim=shared_inter,
            out_dims=[hidden_size],
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_expert_down_proj"),
        )
        self.shared_expert_gate = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[1],
            weight_names=f"mtp.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )

    def _split_q_with_gate(self, weights):
        if self._q_weight_name in weights:
            weight = weights[self._q_weight_name]
            num_heads = self.q_head_num_
            weight = weight.view(num_heads * 2, self.head_dim, -1)
            _q_proj = weight[0::2].reshape(-1, weight.shape[-1])
            _gate_proj = weight[1::2].reshape(-1, weight.shape[-1])
            weights[self._q_weight_name] = _q_proj
            weights[self._o_gate_weight_name] = _gate_proj
