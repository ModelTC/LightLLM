import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import GEMMANormWeight, ROWMMWeight


class Gemma_2bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_qkv(self):
        self.q_proj = ROWMMWeight(
            weight_names=self._q_weight_name,
            data_type=self.data_type_,
            bias_names=self._q_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="q_proj",
        )
        self.kv_proj = ROWMMWeight(
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._k_bias_name, self._v_bias_name],
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="kv_proj",
        )

    def _init_norm(self):
        self.att_norm_weight_ = GEMMANormWeight(self._att_norm_weight_name, self.data_type_)
        self.ffn_norm_weight_ = GEMMANormWeight(self._ffn_norm_weight_name, self.data_type_)
