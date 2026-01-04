import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, NoTpNormWeight


class Starcoder2PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)

        self.wte_weight_ = EmbeddingWeight(
            weight_name="model.embed_tokens.weight",
            data_type=self.data_type_,
        )
        tie_word_embeddings = self.network_config_.get("tie_word_embeddings", False)
        if tie_word_embeddings:
            self.lm_head_weight_: LMHeadWeight = self.wte_weight_
        else:
            self.lm_head_weight_ = LMHeadWeight(
                weight_name="lm_head.weight",
                data_type=self.data_type_,
            )

        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="model.norm.weight",
            data_type=self.data_type_,
            bias_name="model.norm.bias",
        )
        return
