import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight


def generate_alibi(n_head, dtype=torch.float16):
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


class BloomTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _parse_config(self):
        self.n_embed = self.network_config_["n_embed"]
        self.n_head = self.network_config_["num_attention_heads"]
        self.n_inter = self.network_config_["n_embed"] * 4
        self.n_kv_head = self.network_config_["num_attention_heads"]
        self.head_dim = self.network_config_.get("head_dim", self.n_embed // self.n_head)
        # 计算生成alibi
        assert self.n_head % self.tp_world_size_ == 0
        tp_head_num = self.n_head // self.tp_world_size_
        tmp_alibi = generate_alibi(self.n_head, dtype=torch.float32)
        self.tp_alibi = tmp_alibi[self.tp_rank_ * tp_head_num : (self.tp_rank_ + 1) * tp_head_num].contiguous().cuda()

    def _init_weight_names(self):
        self._q_weight_name = f"h.{self.layer_num_}.self_attention.q_proj.weight"
        self._q_bias_name = f"h.{self.layer_num_}.self_attention.q_proj.bias"
        self._k_weight_name = f"h.{self.layer_num_}.self_attention.k_proj.weight"
        self._k_bias_name = f"h.{self.layer_num_}.self_attention.k_proj.bias"
        self._v_weight_name = f"h.{self.layer_num_}.self_attention.v_proj.weight"
        self._v_bias_name = f"h.{self.layer_num_}.self_attention.v_proj.bias"
        self._o_weight_name = f"h.{self.layer_num_}.self_attention.dense.weight"
        self._o_bias_name = f"h.{self.layer_num_}.self_attention.dense.bias"

        self._gate_up_weight_name = f"h.{self.layer_num_}.mlp.dense_h_to_4h.weight"
        self._gate_up_bias_name = f"h.{self.layer_num_}.mlp.dense_h_to_4h.bias"
        self._down_weight_name = f"h.{self.layer_num_}.mlp.dense_4h_to_h.weight"
        self._down_bias_name = f"h.{self.layer_num_}.mlp.dense_4h_to_h.bias"

        self._att_norm_weight_name = f"h.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = f"h.{self.layer_num_}.input_layernorm.bias"
        self._ffn_norm_weight_name = f"h.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = f"h.{self.layer_num_}.post_attention_layernorm.bias"

    def _preprocess_weight(self, weights):
        qkv_weight_name = f"h.{self.layer_num_}.self_attention.query_key_value.weight"
        if qkv_weight_name in weights:
            att_qkv_dense_weight = weights[qkv_weight_name].reshape(self.n_head, 3, -1, self.n_embed)
            weights[self._q_weight_name] = att_qkv_dense_weight[:, 0, :, :].reshape(-1, self.n_embed)
            weights[self._k_weight_name] = att_qkv_dense_weight[:, 1, :, :].reshape(-1, self.n_embed)
            weights[self._v_weight_name] = att_qkv_dense_weight[:, 2, :, :].reshape(-1, self.n_embed)
            del weights[qkv_weight_name]

        qkv_bias_name = f"h.{self.layer_num_}.self_attention.query_key_value.bias"
        if qkv_bias_name in weights:
            att_qkv_dense_bias = weights[qkv_bias_name].reshape(self.n_head, 3, -1)
            weights[self._q_bias_name] = att_qkv_dense_bias[:, 0, :].reshape(-1)
            weights[self._k_bias_name] = att_qkv_dense_bias[:, 1, :].reshape(-1)
            weights[self._v_bias_name] = att_qkv_dense_bias[:, 2, :].reshape(-1)
            del weights[qkv_bias_name]

    def load_hf_weights(self, weights):
        self._preprocess_weight(weights)
        super().load_hf_weights(weights)
        return

    def _init_ffn(self):
        self.gate_up_proj = ROWMMWeight(
            weight_names=self._gate_up_weight_name,
            data_type=self.data_type_,
            bias_names=self._gate_up_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="gate_up_proj",
        )
        self.down_proj = COLMMWeight(
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=self._down_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="down_proj",
        )
