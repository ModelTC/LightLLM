import torch
import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, NoTpNormWeight


class QwenPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.wte_weight_ = EmbeddingWeight(
            weight_name="transformer.wte.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            weight_name="lm_head.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="transformer.ln_f.weight",
            data_type=self.data_type_,
        )
        return
