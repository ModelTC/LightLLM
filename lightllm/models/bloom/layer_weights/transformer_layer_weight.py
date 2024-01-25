import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight


class BloomTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def init_static_params(self):
        # 计算生成alibi
        head_num = self.network_config_["num_attention_heads"]
        tp_head_num = head_num // self.world_size_
        tmp_alibi = self._generate_alibi(head_num, dtype=torch.float32)
        assert head_num % self.world_size_ == 0
        self.tp_alibi = tmp_alibi[self.tp_rank_ * tp_head_num : (self.tp_rank_ + 1) * tp_head_num].contiguous().cuda()
        return

    def load_hf_weights(self, weights):

        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.att_norm_bias_,
            self.q_weight_,
            self.kv_weight_,
            self.q_bias_,
            self.kv_bias_,
            self.o_weight_,
            self.o_bias_,
            self.ffn_norm_weight_,
            self.ffn_norm_bias_,
            self.ffn_1_weight_,
            self.ffn_1_bias_,
            self.ffn_2_weight_,
            self.ffn_2_bias_,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"h.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"h.{self.layer_num_}.input_layernorm.weight"])
        if f"h.{self.layer_num_}.input_layernorm.bias" in weights:
            self.att_norm_bias_ = self._cuda(weights[f"h.{self.layer_num_}.input_layernorm.bias"])

        if f"h.{self.layer_num_}.self_attention.query_key_value.weight" in weights:
            # attention params
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            head_num = self.network_config_["num_attention_heads"]
            att_qkv_dense_weight = weights[f"h.{self.layer_num_}.self_attention.query_key_value.weight"].reshape(
                head_num, 3, -1, n_embed
            )
            self.q_weight_ = self._cuda(
                att_qkv_dense_weight[:, 0, :, :]
                .reshape(-1, n_embed)[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
                .transpose(0, 1)
            )
            self.k_weight_ = (
                att_qkv_dense_weight[:, 1, :, :]
                .reshape(-1, n_embed)[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
                .transpose(0, 1)
            )
            self.v_weight_ = (
                att_qkv_dense_weight[:, 2, :, :]
                .reshape(-1, n_embed)[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
                .transpose(0, 1)
            )

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        if f"h.{self.layer_num_}.self_attention.query_key_value.bias" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            head_num = self.network_config_["num_attention_heads"]
            att_qkv_dense_bias = weights[f"h.{self.layer_num_}.self_attention.query_key_value.bias"].reshape(
                head_num, 3, -1
            )
            self.q_bias_ = self._cuda(
                att_qkv_dense_bias[:, 0, :].reshape(-1)[
                    split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
                ]
            )
            self.k_bias_ = att_qkv_dense_bias[:, 1, :].reshape(-1)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
            ]
            self.v_bias_ = att_qkv_dense_bias[:, 2, :].reshape(-1)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
            ]

        self._try_cat_to(["k_bias_", "v_bias_"], "kv_bias_", cat_dim=0)

        if f"h.{self.layer_num_}.self_attention.dense.weight" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            self.o_weight_ = self._cuda(
                weights[f"h.{self.layer_num_}.self_attention.dense.weight"][
                    :, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
                ].transpose(0, 1)
            )
        if f"h.{self.layer_num_}.self_attention.dense.bias" in weights:
            self.o_bias_ = self._cuda(weights[f"h.{self.layer_num_}.self_attention.dense.bias"])
        return

    def _load_ffn_weights(self, weights):
        if f"h.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            # post attention layernorm params
            self.ffn_norm_weight_ = self._cuda(weights[f"h.{self.layer_num_}.post_attention_layernorm.weight"])
            self.ffn_norm_bias_ = self._cuda(weights[f"h.{self.layer_num_}.post_attention_layernorm.bias"])

        # ffn params
        if f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_1_weight_ = weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight"]
            self.ffn_1_weight_ = self._cuda(
                self.ffn_1_weight_[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :].transpose(
                    0, 1
                )
            )

        if f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_1_bias_ = self._cuda(
                weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias"][
                    split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)
                ]
            )
        if f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_2_weight_ = weights[f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight"]
            self.ffn_2_weight_ = self._cuda(
                self.ffn_2_weight_[:, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)].transpose(
                    0, 1
                )
            )

        if f"h.{self.layer_num_}.mlp.dense_4h_to_h.bias" in weights:
            self.ffn_2_bias_ = self._cuda(weights[f"h.{self.layer_num_}.mlp.dense_4h_to_h.bias"])
        return

    def _generate_alibi(self, n_head, dtype=torch.float16):
        """
        This method is originally the `build_alibi_tensor` function
        in `transformers/models/bloom/modeling_bloom.py`
        of the huggingface/transformers GitHub repository.

        Copyright 2023 ModelTC Team
        Copyright 2022 HuggingFace Inc. team and BigScience workshop

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
        """

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.Tensor(get_slopes(n_head))
        head_alibi = slopes.to(dtype)
        return head_alibi
