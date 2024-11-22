import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import GEMMANormWeight, ROWMMWeight, MultiROWMMWeightNoTP


class Gemma_2bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _init_qkv(self):
        q_split_n_embed = self.head_dim * self.n_head // self.world_size_
        kv_split_n_embed = self.head_dim * self.n_kv_head
        self.q_proj = ROWMMWeight(self._q_weight_name, self.data_type_, q_split_n_embed, bias_name=self._q_bias_name)
        self.kv_proj = MultiROWMMWeightNoTP(
            [self._k_weight_name, self._v_weight_name],
            self.data_type_,
            kv_split_n_embed,
            bias_names=[self._k_bias_name, self._v_bias_name],
        )

    def _init_norm(self):
        self.att_norm_weight_ = GEMMANormWeight(self._att_norm_weight_name, self.data_type_)
        self.ffn_norm_weight_ = GEMMANormWeight(self._ffn_norm_weight_name, self.data_type_)
