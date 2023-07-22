import torch
import math
import numpy as np
from .base_layer_weight import BaseLayerWeight


class TransformerLayerWeight(BaseLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config):
        self.layer_num_ = layer_num
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.data_type_ = data_type
        self.network_config_ = network_config

    def load_ft_weights(self, weight_dir=None):
        # input layernorm params
        self.input_layernorm_weight_ = self.load_to_torch(f"{weight_dir}/model.layers.{self.layer_num_}.input_layernorm.weight.bin").cuda()
        self.input_layernorm_bias_ = self.load_to_torch(f"{weight_dir}/model.layers.{self.layer_num_}.input_layernorm.bias.bin").cuda()

        # attention params
        n_embed = self.network_config_["n_embed"]
        att_qkv_dense_weight = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.attention.query_key_value.weight.{self.tp_rank_}.bin").reshape(n_embed, 3, n_embed // self.world_size_)
        self.q_weight_ = att_qkv_dense_weight[:, 0, :].contiguous().cuda()  # (n_embed, n_embed // self.world_size_)
        self.k_weight_ = att_qkv_dense_weight[:, 1, :].contiguous().cuda()
        self.v_weight_ = att_qkv_dense_weight[:, 2, :].contiguous().cuda()

        att_qkv_dense_bias = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.attention.query_key_value.bias.{self.tp_rank_}.bin").reshape(3, n_embed // self.world_size_)
        self.q_bias_ = att_qkv_dense_bias[0, :].contiguous().cuda()
        self.k_bias_ = att_qkv_dense_bias[1, :].contiguous().cuda()
        self.v_bias_ = att_qkv_dense_bias[2, :].contiguous().cuda()

        # attention output dense params
        self.att_out_dense_weight_ = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.attention.dense.weight.{self.tp_rank_}.bin").reshape(n_embed // self.world_size_, n_embed).contiguous().cuda()
        self.att_out_dense_bias_ = self.load_to_torch(f"{weight_dir}/model.layers.{self.layer_num_}.attention.dense.bias.bin").cuda()

        # post attention layernorm params
        self.post_attention_layernorm_weight_ = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.post_attention_layernorm.weight.bin").cuda()
        self.post_attention_layernorm_bias_ = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.post_attention_layernorm.bias.bin").cuda()

        # ffn params
        self.ffn_1_weight_ = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.mlp.dense_h_to_4h.weight.{self.tp_rank_}.bin").reshape(n_embed, n_embed * 4 // self.world_size_).cuda()
        self.ffn_1_bias_ = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.mlp.dense_h_to_4h.bias.{self.tp_rank_}.bin").cuda()

        self.ffn_2_weight_ = self.load_to_torch(
            f"{weight_dir}/model.layers.{self.layer_num_}.mlp.dense_4h_to_h.weight.{self.tp_rank_}.bin").reshape(n_embed * 4 // self.world_size_, n_embed).cuda()
        self.ffn_2_bias_ = self.load_to_torch(f"{weight_dir}/model.layers.{self.layer_num_}.mlp.dense_4h_to_h.bias.bin").cuda()

        # 计算生成alibi
        head_num = self.network_config_["num_attention_heads"]
        tp_head_num = head_num // self.world_size_
        tmp_alibi = self.generate_alibi(head_num, dtype=torch.float32)
        assert head_num % self.world_size_ == 0
        self.tp_alibi = tmp_alibi[self.tp_rank_ * tp_head_num: (self.tp_rank_ + 1) * tp_head_num].contiguous().cuda()

    def load_hf_weights(self, weights):
        if isinstance(self.data_type_, str):
            if self.data_type_ == "fp16":
                self.data_type_ = torch.float16
            elif self.data_type_ == "fp32":
                self.data_type_ = torch.float32
            else:
                raise
                # input layernorm params
        if f"h.{self.layer_num_}.input_layernorm.weight" in weights:
            self.input_layernorm_weight_ = weights[f"h.{self.layer_num_}.input_layernorm.weight"].contiguous().cuda()
        if f"h.{self.layer_num_}.input_layernorm.bias" in weights:
            self.input_layernorm_bias_ = weights[f"h.{self.layer_num_}.input_layernorm.bias"].contiguous().cuda()

        if f"h.{self.layer_num_}.self_attention.query_key_value.weight" in weights:
            # attention params
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            head_num = self.network_config_["num_attention_heads"]
            att_qkv_dense_weight = weights[f"h.{self.layer_num_}.self_attention.query_key_value.weight"].reshape(head_num, 3, -1, n_embed)
            self.q_weight_ = att_qkv_dense_weight[:,
                                                  0,
                                                  :,
                                                  :].reshape(-1,
                                                             n_embed)[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1),
                                                                      :].transpose(0,
                                                                                   1).contiguous().cuda()
            self.k_weight_ = att_qkv_dense_weight[:,
                                                  1,
                                                  :,
                                                  :].reshape(-1,
                                                             n_embed)[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1),
                                                                      :].transpose(0,
                                                                                   1).contiguous().cuda()
            self.v_weight_ = att_qkv_dense_weight[:,
                                                  2,
                                                  :,
                                                  :].reshape(-1,
                                                             n_embed)[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1),
                                                                      :].transpose(0,
                                                                                   1).contiguous().cuda()
        if f"h.{self.layer_num_}.self_attention.query_key_value.bias" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            head_num = self.network_config_["num_attention_heads"]
            att_qkv_dense_bias = weights[f"h.{self.layer_num_}.self_attention.query_key_value.bias"].reshape(head_num, 3, -1)
            self.q_bias_ = att_qkv_dense_bias[:, 0, :].reshape(-1)[split_n_embed *
                                                                   self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].contiguous().cuda()
            self.k_bias_ = att_qkv_dense_bias[:, 1, :].reshape(-1)[split_n_embed *
                                                                   self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].contiguous().cuda()
            self.v_bias_ = att_qkv_dense_bias[:, 2, :].reshape(-1)[split_n_embed *
                                                                   self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].contiguous().cuda()

        if f"h.{self.layer_num_}.self_attention.dense.weight" in weights:
            n_embed = self.network_config_["n_embed"]
            split_n_embed = n_embed // self.world_size_
            self.att_out_dense_weight_ = weights[f"h.{self.layer_num_}.self_attention.dense.weight"][:,
                                                                                                     split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].transpose(0, 1).contiguous().cuda()
        if f"h.{self.layer_num_}.self_attention.dense.bias" in weights:
            self.att_out_dense_bias_ = weights[f"h.{self.layer_num_}.self_attention.dense.bias"].contiguous().cuda()

        if f"h.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            # post attention layernorm params
            self.post_attention_layernorm_weight_ = weights[f"h.{self.layer_num_}.post_attention_layernorm.weight"].contiguous().cuda()
            self.post_attention_layernorm_bias_ = weights[f"h.{self.layer_num_}.post_attention_layernorm.bias"].contiguous().cuda()

        # ffn params
        if f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_1_weight_ = weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight"]
            self.ffn_1_weight_ = self.ffn_1_weight_[split_n_embed * self.tp_rank_: split_n_embed *
                                                    (self.tp_rank_ + 1), :].transpose(0, 1).contiguous().cuda()

        if f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_1_bias_ = weights[f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias"][split_n_embed *
                                                                                      self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].contiguous().cuda()

        if f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight" in weights:
            n_embed = self.network_config_["n_embed"] * 4
            split_n_embed = n_embed // self.world_size_
            self.ffn_2_weight_ = weights[f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight"]
            self.ffn_2_weight_ = self.ffn_2_weight_[:, split_n_embed *
                                                    self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)].transpose(0, 1).contiguous().cuda()

        if f"h.{self.layer_num_}.mlp.dense_4h_to_h.bias" in weights:
            self.ffn_2_bias_ = weights[f"h.{self.layer_num_}.mlp.dense_4h_to_h.bias"].contiguous().cuda()

    def init_hf_alibi(self):
        # 计算生成alibi
        head_num = self.network_config_["num_attention_heads"]
        tp_head_num = head_num // self.world_size_
        tmp_alibi = self.generate_alibi(head_num, dtype=torch.float32)
        assert head_num % self.world_size_ == 0
        self.tp_alibi = tmp_alibi[self.tp_rank_ * tp_head_num: (self.tp_rank_ + 1) * tp_head_num].contiguous().cuda()

    def generate_alibi(self, n_head, dtype=torch.float16):
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
                return [start * ratio**i for i in range(n)]

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
