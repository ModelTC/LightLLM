import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class ChatGLM2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        assert network_config["num_attention_heads"] % self.world_size_ == 0

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_weight_,
            self.q_bias_,
            self.kv_bias_,
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
        if f"transformer.encoder.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.input_layernorm.weight"]
            self.att_norm_weight_ = self._cuda(self.att_norm_weight_)

        # attention params
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        multi_query_group_num = self.network_config_["multi_query_group_num"]
        tp_kv_head_dim = multi_query_group_num // self.world_size_ * head_dim
        split_n_embed = n_embed // self.world_size_
        if f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.weight" in weights:
            qkv_weight_ = (
                weights[f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.weight"]
                .transpose(0, 1)
                .contiguous()
                .to(self.data_type_)
            )
            self.q_weight_ = qkv_weight_[:, :n_embed][
                :, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
            ]
            self.q_weight_ = self._cuda(self.q_weight_)
            k_weight_ = qkv_weight_[:, n_embed : n_embed + head_dim * multi_query_group_num]
            self.k_weight_ = k_weight_[:, tp_kv_head_dim * self.tp_rank_ : tp_kv_head_dim * (self.tp_rank_ + 1)]

            v_weight_ = qkv_weight_[
                :, n_embed + multi_query_group_num * head_dim : n_embed + 2 * multi_query_group_num * head_dim
            ]
            self.v_weight_ = v_weight_[:, tp_kv_head_dim * self.tp_rank_ : tp_kv_head_dim * (self.tp_rank_ + 1)]

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        if f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.bias" in weights:

            qkv_bias_ = weights[f"transformer.encoder.layers.{self.layer_num_}.self_attention.query_key_value.bias"].to(
                self.data_type_
            )
            self.q_bias_ = qkv_bias_[:n_embed][split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            self.q_bias_ = self._cuda(self.q_bias_)
            k_bias_ = qkv_bias_[n_embed : n_embed + head_dim * multi_query_group_num]
            self.k_bias_ = k_bias_[tp_kv_head_dim * self.tp_rank_ : tp_kv_head_dim * (self.tp_rank_ + 1)]
            v_bias_ = qkv_bias_[
                n_embed + multi_query_group_num * head_dim : n_embed + 2 * multi_query_group_num * head_dim
            ]
            self.v_bias_ = v_bias_[tp_kv_head_dim * self.tp_rank_ : tp_kv_head_dim * (self.tp_rank_ + 1)]

        self._try_cat_to(["k_bias_", "v_bias_"], "kv_bias_", cat_dim=0)

        # attention output dense params
        if f"transformer.encoder.layers.{self.layer_num_}.self_attention.dense.weight" in weights:
            self.o_weight_ = weights[f"transformer.encoder.layers.{self.layer_num_}.self_attention.dense.weight"]
            self.o_weight_ = self.o_weight_[:, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self.o_weight_.transpose(0, 1)
            self.o_weight_ = self._cuda(self.o_weight_)

    def _load_ffn_weights(self, weights):
        if f"transformer.encoder.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = weights[
                f"transformer.encoder.layers.{self.layer_num_}.post_attention_layernorm.weight"
            ]
            self.ffn_norm_weight_ = self._cuda(self.ffn_norm_weight_)

        # ffn params
        ffn_hidden_size = self.network_config_["ffn_hidden_size"]
        split_inter_size = ffn_hidden_size // self.world_size_
        if f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_h_to_4h.weight" in weights:
            tweights = weights[f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_h_to_4h.weight"].to(
                self.data_type_
            )
            gate_proj = tweights[:ffn_hidden_size, :]
            gate_proj = gate_proj[split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = gate_proj.transpose(0, 1)

            up_proj = tweights[ffn_hidden_size : 2 * ffn_hidden_size, :]
            up_proj = up_proj[split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = up_proj.transpose(0, 1)

            self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=1)

        if f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_4h_to_h.weight" in weights:
            self.down_proj = weights[f"transformer.encoder.layers.{self.layer_num_}.mlp.dense_4h_to_h.weight"].to(
                self.data_type_
            )
            self.down_proj = self.down_proj[
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ].transpose(0, 1)
            self.down_proj = self._cuda(self.down_proj)
        return
