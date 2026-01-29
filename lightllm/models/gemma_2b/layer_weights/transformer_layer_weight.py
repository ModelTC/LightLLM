import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpGEMMANormWeight, ROWMMWeight


class Gemma_2bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_qkv(self):
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        kv_out_dim = self.k_head_num_ * self.head_dim
        self.q_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim],
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_method=self.get_quant_method("q_proj"),
        )
        self.kv_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[kv_out_dim, kv_out_dim],
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("kv_proj"),
        )

    def _init_norm(self):
        self.att_norm_weight_ = NoTpGEMMANormWeight(
            dim=self.n_embed, weight_name=self._att_norm_weight_name, data_type=self.data_type_, bias_name=None
        )
        self.ffn_norm_weight_ = NoTpGEMMANormWeight(
            dim=self.n_embed, weight_name=self._ffn_norm_weight_name, data_type=self.data_type_, bias_name=None
        )
