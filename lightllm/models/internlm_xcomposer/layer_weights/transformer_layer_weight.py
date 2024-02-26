import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight

from lightllm.models.internlm2.layer_weights.transformer_layer_weight import Internlm2TransformerLayerWeight


class InternlmComposerTransformerLayerWeight(Internlm2TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.lora_scaling = 1.
        return

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        self._load_lora_weights(weights)
        return

    def verify_load(self):
        errors = "weights load not ok"

        # handle internlm 20b, which has no bias, so set q k v o bias to zero
        if not self.network_config_.get("bias", True):
            for layer_type in ("q", "kv", "o"):
                attr_name = f"{layer_type}_bias_"
                if hasattr(self, attr_name):
                    continue
                setattr(self, attr_name, self._cuda(torch.zeros(1)))

        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_weight_,
            self.o_weight_,
            self.q_bias_,
            self.kv_bias_,
            self.o_bias_,
            self.ffn_norm_weight_,
            self.gate_up_proj,
            self.down_proj,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_lora_weights(self, weights):
        # input layernorm params
        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = (
            n_embed
            // self.network_config_["num_attention_heads"]
            * self.network_config_["num_key_value_heads"]
            // self.world_size_
        )
        head_dim = n_embed // self.network_config_["num_attention_heads"]
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.attention.wqkv.Plora_A.weight" in weights:
            self.qkv_loraA_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.attention.wqkv.Plora_A.weight"]).transpose(0, 1)

        if f"model.layers.{self.layer_num_}.attention.wqkv.Plora_B.weight" in weights:
            qkv_loraB_weight_ = weights[f"model.layers.{self.layer_num_}.attention.wqkv.Plora_B.weight"]
            q_groups = self.network_config_["num_attention_heads"] // self.network_config_["num_key_value_heads"]
            qkv_loraB_weight_ = qkv_loraB_weight_.reshape(self.network_config_["num_key_value_heads"], q_groups + 2, head_dim, -1)
            q_loraB_weight_ = qkv_loraB_weight_[:, :q_groups, :, :].reshape(-1, qkv_loraB_weight_.shape[-1])
            self.q_loraB_weight_ = self._cuda(
                q_loraB_weight_[q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1) :].transpose(0, 1)
            )

            k_loraB_weight_ = qkv_loraB_weight_[:, -2, :, :].reshape(-1, qkv_loraB_weight_.shape[-1])
            self.k_loraB_weight_ = k_loraB_weight_[
                kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1) :
            ].transpose(0, 1)
            v_loraB_weight_ = qkv_loraB_weight_[:, -1, :, :].reshape(-1, qkv_loraB_weight_.shape[-1])
            self.v_loraB_weight_ = v_loraB_weight_[
                kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1) :
            ].transpose(0, 1)
        
        self._try_cat_to(["k_loraB_weight_", "v_loraB_weight_"], "kv_loraB_weight_", cat_dim=1)

        if f"model.layers.{self.layer_num_}.attention.wo.Plora_A.weight" in weights:
            wo_loraA_weight_ = weights[f"model.layers.{self.layer_num_}.attention.wo.Plora_A.weight"]
            wo_loraA_weight_ = wo_loraA_weight_[:, q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1)]
            self.wo_loraA_weight_ = self._cuda(wo_loraA_weight_).transpose(0, 1)

        if f"model.layers.{self.layer_num_}.attention.wo.Plora_B.weight" in weights:
            self.wo_loraB_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.attention.wo.Plora_B.weight"]).transpose(0, 1)

        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_
        split_start = split_inter_size * self.world_size_
        split_end = split_inter_size * (self.world_size_ + 1)

        if f"model.layers.{self.layer_num_}.feed_forward.w3.Plora_A.weight" in weights:
            self.up_loraA_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.feed_forward.w3.Plora_A.weight"]).transpose(0, 1)
        if f"model.layers.{self.layer_num_}.feed_forward.w3.Plora_B.weight" in weights:
            up_loraB_weight_ = weights[f"model.layers.{self.layer_num_}.feed_forward.w3.Plora_B.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.up_loraB_weight_ = self._cuda(up_loraB_weight_).transpose(0, 1)


        if f"model.layers.{self.layer_num_}.feed_forward.w1.Plora_A.weight" in weights:
            self.gate_loraA_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.feed_forward.w1.Plora_A.weight"]).transpose(0, 1)
        
        if f"model.layers.{self.layer_num_}.feed_forward.w1.Plora_B.weight" in weights:
            gate_loraB_weight_ = weights[f"model.layers.{self.layer_num_}.feed_forward.w1.Plora_B.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.gate_loraB_weight_ = self._cuda(gate_loraB_weight_).transpose(0, 1)

        if f"model.layers.{self.layer_num_}.feed_forward.w2.Plora_A.weight" in weights:
            down_loraA_weight_ = weights[f"model.layers.{self.layer_num_}.feed_forward.w2.Plora_A.weight"][
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.down_loraA_weight_ = self._cuda(down_loraA_weight_).transpose(0, 1)
        if f"model.layers.{self.layer_num_}.feed_forward.w2.Plora_B.weight" in weights:
            self.down_loraB_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.feed_forward.w2.Plora_B.weight"]).transpose(0, 1)
        
