from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, NoTpNormWeight


class CoherePreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        tie_weight = self.network_config_.get("tie_word_embeddings", True)

        self.wte_weight_ = EmbeddingWeight(
            weight_name="model.embed_tokens.weight",
            data_type=self.data_type_,
        )
        if tie_weight:
            self.lm_head_weight_ = self.wte_weight_
        else:
            self.lm_head_weight_ = LMHeadWeight(
                weight_name="model.lm_head.weight",
                data_type=self.data_type_,
            )
        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="model.norm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
