import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MixtralTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.n_routed_experts = network_config["num_local_experts"]
        self.experts_w1 = [None] * self.n_routed_experts
        self.w2_list = [None] * self.n_routed_experts
        self.experts_w3 = [None] * self.n_routed_experts
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
            self.moe_gate,
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
            self.moe_gate = weights[f"model.layers.{self.layer_num_}.block_sparse_moe.gate.weight"]
            self.moe_gate = self._cuda(self.moe_gate.transpose(0, 1))

        for expert_idx in range(self.n_routed_experts):
            expert_w1 = None
            expert_w2 = None
            expert_gate_up_proj = None
            expert_w3 = None

            if f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w1.weight" in weights:
                expert_w1 = weights[f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w1.weight"][
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
                ]
                self.experts_w1[expert_idx] = self._cuda(expert_w1)
            if f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w2.weight" in weights:
                expert_w2 = weights[f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w2.weight"][
                    :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
                ]
                self.w2_list[expert_idx] = self._cuda(expert_w2)

            if f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w3.weight" in weights:
                expert_w3 = weights[f"model.layers.{self.layer_num_}.block_sparse_moe.experts.{expert_idx}.w3.weight"][
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
                ]
                self.experts_w3[expert_idx] = self._cuda(expert_w3)

            with self.lock:
                if (
                    hasattr(self, "experts_w1")
                    and None not in self.experts_w1
                    and None not in self.experts_w3
                    and None not in self.w2_list
                ):

                    w1_list = []
                    for i_experts in range(self.n_routed_experts):
                        expert_gate_up_proj = torch.cat([self.experts_w1[i_experts], self.experts_w3[i_experts]], dim=0)
                        expert_gate_up_proj = self._cuda(expert_gate_up_proj)
                        w1_list.append(expert_gate_up_proj)

                    inter_shape, hidden_size = w1_list[0].shape[0], w1_list[0].shape[1]
                    self.w1 = torch._utils._flatten_dense_tensors(w1_list).view(len(w1_list), inter_shape, hidden_size)
                    inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                    self.w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(
                        len(self.w2_list), inter_shape, hidden_size
                    )

                    delattr(self, "w2_list")
                    delattr(self, "experts_w1")
                    delattr(self, "experts_w3")
        return
