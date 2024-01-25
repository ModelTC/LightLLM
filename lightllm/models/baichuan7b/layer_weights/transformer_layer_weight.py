import torch
import math
import numpy as np

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class BaiChuan7bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for baichuan
        if f"model.layers.{self.layer_num_}.self_attn.W_pack.weight" in weights:
            qkv_weights = weights[f"model.layers.{self.layer_num_}.self_attn.W_pack.weight"]
            split_size = qkv_weights.shape[0] // 3
            q_weights, k_weights, v_weights = torch.split(qkv_weights, split_size, dim=0)
            self.q_weight_ = q_weights[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))
            k_weight_ = k_weights[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = k_weight_.transpose(0, 1)
            v_weight_ = v_weights[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = v_weight_.transpose(0, 1)

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][
                :, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
            ]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))
        return
