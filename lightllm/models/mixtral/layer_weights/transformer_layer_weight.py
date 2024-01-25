import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MixtralTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.experts = [{"w1": None, "w2": None, "w3": None} for _ in range(self.network_config_["num_local_experts"])]
        return

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_weight_,
            self.o_weight_,
            self.ffn_norm_weight_,
            self.gate,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, str(i) + " " + str(self.layer_num_) + " " + errors
        for i in range(self.network_config_["num_local_experts"]):
            assert self.experts[i]["w1"] is not None, (
                "layer " + str(self.layer_num_) + " expert " + str(i) + " w1 " + errors
            )
            assert self.experts[i]["w2"] is not None, (
                "layer " + str(self.layer_num_) + " expert " + str(i) + " w2 " + errors
            )
            assert self.experts[i]["w3"] is not None, (
                "layer " + str(self.layer_num_) + " expert " + str(i) + " w3 " + errors
            )
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
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"]
            self.q_weight_ = self.q_weight_[q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"]
            k_weight_ = k_weight_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = k_weight_.transpose(0, 1)

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"]
            v_weight_ = v_weight_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = v_weight_.transpose(0, 1)

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

        if f"model.layers.{self.layer_num_}.block_sparse_moe.gate.weight" in weights:
            self.gate = weights[f"model.layers.{self.layer_num_}.block_sparse_moe.gate.weight"]
            self.gate = self._cuda(self.gate.transpose(0, 1))

        for expert_idx in range(self.network_config_["num_local_experts"]):
            if f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w1.weight" in weights:
                self.experts[expert_idx]["w1"] = weights[
                    f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w1.weight"
                ][split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
                self.experts[expert_idx]["w1"] = self._cuda(self.experts[expert_idx]["w1"].transpose(0, 1))
            if f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w2.weight" in weights:
                self.experts[expert_idx]["w2"] = weights[
                    f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w2.weight"
                ][:, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)]
                self.experts[expert_idx]["w2"] = self._cuda(self.experts[expert_idx]["w2"].transpose(0, 1))

            if f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w3.weight" in weights:
                self.experts[expert_idx]["w3"] = weights[
                    f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w3.weight"
                ][split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
                self.experts[expert_idx]["w3"] = self._cuda(self.experts[expert_idx]["w3"].transpose(0, 1))
        return
