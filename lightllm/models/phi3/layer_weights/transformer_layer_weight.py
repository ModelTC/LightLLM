import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Phi3TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def load_hf_weights(self, weights):
        qkv_weight_name = f"model.layers.{self.layer_num_}.self_attn.qkv_proj.weight"
        if qkv_weight_name in weights:
            qkv_weight_ = weights[qkv_weight_name]
            n_embed = self.network_config_["hidden_size"]
            kv_n_embed = (
                n_embed // self.network_config_["num_attention_heads"] * self.network_config_["num_key_value_heads"]
            )
            q_weight_ = qkv_weight_[:n_embed, :]
            k_weight_ = qkv_weight_[n_embed : n_embed + kv_n_embed, :]
            v_weight_ = qkv_weight_[n_embed + kv_n_embed : n_embed + 2 * kv_n_embed, :]
            weights[self._q_weight_name] = q_weight_
            weights[self._k_weight_name] = k_weight_
            weights[self._v_weight_name] = v_weight_
            del weights[qkv_weight_name]

        gate_up_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"
        if gate_up_weight_name in weights:
            inter_size = self.network_config_["intermediate_size"]
            gate_up_proj = weights[gate_up_weight_name]
            gate_weight_ = gate_up_proj[0:inter_size, :]
            up_weight_ = gate_up_proj[inter_size:, :]
            weights[self._gate_weight_name] = gate_weight_
            weights[self._up_weight_name] = up_weight_
            del weights[gate_up_weight_name]
        super().load_hf_weights(weights)
