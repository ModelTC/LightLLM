import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    RMSNormWeight,
    LayerNormWeight,
    TpRMSNormWeight,
)
from lightllm.utils.dist_utils import get_current_device_id


class ViTTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _cuda(self, cpu_tensor):
        device_id = get_current_device_id()
        return cpu_tensor.contiguous().to(self.data_type_).cuda(device_id)

    def _parse_config(self):
        self.padding_hidden_size = self.network_config_["padding_hidden_size"]
        self.qk_norm = self.network_config_["qk_normalization"]
        self.use_ls = self.network_config_.get("use_ls", False)
        self.qkv_bias = self.network_config_.get("qkv_bias", True)
        self.layer_norm_eps = self.network_config_.get("layer_norm_eps", 1e-6)
        self.norm_type = self.network_config_.get("norm_type", "layer_norm")
        self.n_embed = self.network_config_["hidden_size"] + self.padding_hidden_size
        mlp_ratio = self.network_config_.get("mlp_ratio", 4)
        self.n_inter = self.network_config_.get("intermediate_size", int(self.n_embed * mlp_ratio))

    def _init_weight_names(self):
        self._att_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight"

        self._q_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.q.weight"
        self._k_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.k.weight"
        self._v_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.v.weight"

        if self.qkv_bias:
            self._q_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.q.bias"
            self._k_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.k.bias"
            self._v_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.v.bias"
        else:
            self._q_bias_name = None
            self._k_bias_name = None
            self._v_bias_name = None

        self._o_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.weight"
        self._o_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.bias"

        self.fc1_weight_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.weight"
        self.fc1_bias_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.bias"
        self.fc2_weight_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.weight"
        self.fc2_bias_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.bias"

        self._ls1_name = f"vision_model.encoder.layers.{self.layer_num_}.ls1"
        self._ls2_name = f"vision_model.encoder.layers.{self.layer_num_}.ls2"

        self._att_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight"
        self._ffn_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight"

        if self.norm_type == "layer_norm":
            self._att_norm_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.norm1.bias"
            self._ffn_norm_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.norm2.bias"
        else:
            self._att_norm_bias_name = None
            self._ffn_norm_bias_name = None

        if self.qk_norm:
            self._q_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight"
            self._k_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight"
            self._q_norm_bias_name = None
            self._k_norm_bias_name = None

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()

    def _init_qkv(self):
        self.qkv_proj = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_embed, self.n_embed, self.n_embed],
            weight_names=[self._q_weight_name, self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._q_bias_name, self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("qkv_proj"),
        )

    def _init_o(self):
        self.o_proj = COLMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_embed],
            weight_names=self._o_weight_name,
            data_type=self.data_type_,
            bias_names=self._o_bias_name,
            quant_method=self.get_quant_method("o_proj"),
        )

    def _init_ffn(self):
        self.ffn_1_proj_ = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_inter],
            weight_names=self.fc1_weight_name_,
            data_type=self.data_type_,
            bias_names=self.fc1_bias_name_,
            quant_method=self.get_quant_method("ffn_1_proj"),
        )

        self.ffn_2_proj_ = COLMMWeight(
            in_dim=self.n_inter,
            out_dims=[self.n_embed],
            weight_names=self.fc2_weight_name_,
            data_type=self.data_type_,
            bias_names=self.fc2_bias_name_,
            quant_method=self.get_quant_method("ffn_2_proj"),
        )

    def _init_norm(self):
        norm_weight_cls = RMSNormWeight if self.norm_type == "rms_norm" else LayerNormWeight
        hidden_size = self.network_config_["hidden_size"]
        self.att_norm_weight_ = norm_weight_cls(
            dim=hidden_size,
            weight_name=self._att_norm_weight_name,
            data_type=self.data_type_,
            bias_name=self._att_norm_bias_name,
        )
        self.ffn_norm_weight_ = norm_weight_cls(
            dim=hidden_size,
            weight_name=self._ffn_norm_weight_name,
            data_type=self.data_type_,
            bias_name=self._ffn_norm_bias_name,
        )
        if self.qk_norm:
            head_num = self.network_config_["num_attention_heads"]
            head_dim = self.network_config_["hidden_size"] // head_num
            head_dim = self.network_config_.get("head_dim", head_dim)
            self.q_norm_weight_ = TpRMSNormWeight(
                head_num=head_num,
                head_dim=head_dim,
                weight_name=self._q_norm_weight_name,
                data_type=self.data_type_,
            )
            self.k_norm_weight_ = TpRMSNormWeight(
                head_num=head_num,
                head_dim=head_dim,
                weight_name=self._k_norm_weight_name,
                data_type=self.data_type_,
            )

    def load_hf_weights(self, weights):
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight" in weights:
            n_embed = self.network_config_["hidden_size"]
            att_qkv_dense_weight = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight"]
            att_qkv_dense_weight = att_qkv_dense_weight.reshape(3, n_embed, -1)
            q_weight_ = F.pad(att_qkv_dense_weight[0, :, :], (0, 0, 0, self.padding_hidden_size))
            k_weight_ = F.pad(att_qkv_dense_weight[1, :, :], (0, 0, 0, self.padding_hidden_size))
            v_weight_ = F.pad(att_qkv_dense_weight[2, :, :], (0, 0, 0, self.padding_hidden_size))
            del weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight"]
            weights[self._q_weight_name] = q_weight_
            weights[self._k_weight_name] = k_weight_
            weights[self._v_weight_name] = v_weight_

        if self._o_weight_name in weights:
            weights[self._o_weight_name] = F.pad(weights[self._o_weight_name], (0, self.padding_hidden_size, 0, 0))

        if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias" in weights:
            n_embed = self.network_config_["hidden_size"]
            att_qkv_dense_bias = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias"]
            att_qkv_dense_bias = F.pad(att_qkv_dense_bias, (0, self.padding_hidden_size)).reshape(3, -1)
            q_bias_ = att_qkv_dense_bias[0]
            k_bias_ = att_qkv_dense_bias[1]
            v_bias_ = att_qkv_dense_bias[2]
            weights[self._q_bias_name] = q_bias_
            weights[self._k_bias_name] = k_bias_
            weights[self._v_bias_name] = v_bias_
            del weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias"]

        if f"vision_model.encoder.layers.{self.layer_num_}.ls1" in weights:
            ls1 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls1"]
            self.ls1 = self._cuda(ls1)

        if f"vision_model.encoder.layers.{self.layer_num_}.ls2" in weights:
            ls2 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls2"]
            self.ls2 = self._cuda(ls2)
            self.use_ls = True

        return super().load_hf_weights(weights)
