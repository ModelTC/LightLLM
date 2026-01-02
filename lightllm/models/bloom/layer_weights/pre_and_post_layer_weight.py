import torch
import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, NoTpNormWeight


class BloomPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.pre_norm_weight_ = NoTpNormWeight(
            weight_name="word_embeddings_layernorm.weight",
            data_type=self.data_type_,
            bias_name="word_embeddings_layernorm.bias",
        )
        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="ln_f.weight",
            data_type=self.data_type_,
            bias_name="ln_f.bias",
        )

        self.wte_weight_ = EmbeddingWeight(
            weight_name="word_embeddings.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = self.wte_weight_
