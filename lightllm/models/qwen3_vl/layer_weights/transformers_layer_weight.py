import os
import torch
import math
import numpy as np
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    MultiROWMMWeight,
    COLMMWeight,
    NormWeight,
    FusedMoeWeightTP,
    FusedMoeWeightEP,
    ROWBMMWeight,
)


class Qwen3VLTransformerLayerWeight(Qwen3TransformerLayerWeight):  # 后面看要不要改
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._q_bias_name = None
        self._k_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._k_bias_name = None
        self._v_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._kv_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.kv_proj.weight"
        self._kv_bias_name = None
        self._o_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._att_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _parse_config(self):
        self.tp_q_head_num_ = self.network_config_["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = max(self.network_config_["num_key_value_heads"] // self.tp_world_size_, 1)
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        self.head_dim = self.network_config_.get("head_dim", head_dim)
        assert (self.tp_k_head_num_ * self.tp_world_size_) % self.network_config_["num_key_value_heads"] == 0

    def _repeat_weight(self, name, weights):
        # for tp_world_size_ > num_key_value_heads
        if name not in weights:
            return

        tensor = weights[name]
        num_kv_heads = self.network_config_["num_key_value_heads"]
        repeat_size = (self.tp_k_head_num_ * self.tp_world_size_) // num_kv_heads

        if tensor.ndim == 1:
            # Bias (1D tensor)
            tensor = tensor.reshape(num_kv_heads, -1).unsqueeze(1).repeat(1, repeat_size, 1).reshape(-1)
        else:
            # Weight (2D tensor)
            tensor = (
                tensor.reshape(num_kv_heads, -1, tensor.shape[-1])
                .unsqueeze(1)
                .repeat(1, repeat_size, 1, 1)
                .reshape(-1, tensor.shape[-1])
            )
        weights[name] = tensor

    def load_hf_weights(self, weights):
        self._repeat_weight(self._k_weight_name, weights)
        self._repeat_weight(self._v_weight_name, weights)
        if self._k_bias_name is not None and self._v_bias_name is not None:
            self._repeat_weight(self._k_bias_name, weights)
            self._repeat_weight(self._v_bias_name, weights)
        return super().load_hf_weights(weights)


class Qwen3VLMOETransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._q_bias_name = None
        self._k_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._k_bias_name = None
        self._v_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._kv_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.kv_proj.weight"
        self._kv_bias_name = None
        self._o_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._att_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            weight_name=f"model.language_model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=self.data_type_,
            layer_num=self.layer_num_,
            name="moe_gate",
            tp_rank=0,
            tp_world_size=1,
        )
        moe_mode = os.getenv("MOE_MODE", "TP")
        assert moe_mode in ["EP", "TP"]

        if moe_mode == "TP":
            self.experts = FusedMoeWeightTP(
                gate_proj_name="gate_up_proj",
                down_proj_name="down_proj",
                up_proj_name=None,
                e_score_correction_bias_name="",
                weight_prefix=f"model.language_model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                split_inter_size=moe_intermediate_size // self.tp_world_size_,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
                num_fused_shared_experts=0,
            )
        elif moe_mode == "EP":
            self.experts = FusedMoeWeightEP(
                gate_proj_name="gate_up_proj",
                down_proj_name="down_proj",
                up_proj_name=None,
                e_score_correction_bias_name="",
                weight_prefix=f"model.language_model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
            )
        else:
            raise ValueError(f"Unsupported moe mode: {moe_mode}")
