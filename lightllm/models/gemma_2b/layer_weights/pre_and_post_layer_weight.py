from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, NoTpGEMMANormWeight


class Gemma_2bPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]

        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="model.embed_tokens.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = self.wte_weight_

        self.final_norm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="model.norm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )

        return
