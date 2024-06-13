import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Phi3TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_weight_,
            self.o_weight_,
            self.ffn_norm_weight_,
            self.gate_up_proj,
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
        kv_n_embed = (
            n_embed // self.network_config_["num_attention_heads"] * self.network_config_["num_key_value_heads"]
        )
        kv_split_n_embed = (
            n_embed
            // self.network_config_["num_attention_heads"]
            * self.network_config_["num_key_value_heads"]
            // self.world_size_
        )
        if f"model.layers.{self.layer_num_}.self_attn.qkv_proj.weight" in weights:
            qkv_weight_ = (
                weights[f"model.layers.{self.layer_num_}.self_attn.qkv_proj.weight"]
                .transpose(0, 1)
                .contiguous()
                .to(self.data_type_)
            )
            self.q_weight_ = qkv_weight_[:, :n_embed][
                :, q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1)
            ]
            self.q_weight_ = self._cuda(self.q_weight_)
            k_weight_ = qkv_weight_[:, n_embed : n_embed + kv_n_embed]
            self.k_weight_ = k_weight_[:, kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1)]

            v_weight_ = qkv_weight_[:, n_embed + kv_n_embed : n_embed + 2 * kv_n_embed]
            self.v_weight_ = v_weight_[:, kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1)]

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            self.o_weight_ = self.o_weight_[:, q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))

        return

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(
                weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"]
            )

        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight" in weights:
            gate_up_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"]
            gate_proj = gate_up_proj[0:inter_size][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.gate_proj = gate_proj.transpose(0, 1)

            up_proj = gate_up_proj[inter_size:][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.up_proj = up_proj.transpose(0, 1)

        self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=1)

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))
        return
