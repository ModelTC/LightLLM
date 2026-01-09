from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, NoTpGEMMANormWeight


class Gemma3PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)

        self.wte_weight_ = EmbeddingWeight(
            weight_name="language_model.model.embed_tokens.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = self.wte_weight_

        self.final_norm_weight_ = NoTpGEMMANormWeight(
            weight_name="language_model.model.norm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        return
