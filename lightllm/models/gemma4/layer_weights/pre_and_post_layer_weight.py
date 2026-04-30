from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
)


class Gemma4PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]

        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="model.language_model.embed_tokens.weight",
            data_type=self.data_type_,
        )
        # lm_head is tied to input embedding for Gemma-4 (no separate lm_head.weight).
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=self.data_type_,
            embedding_weight=self.wte_weight_,
        )

        # Gemma-4 uses standard RMSNorm (not the gemma2/3 (1+w) variant).
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.language_model.norm.weight",
            data_type=self.data_type_,
        )
        return
