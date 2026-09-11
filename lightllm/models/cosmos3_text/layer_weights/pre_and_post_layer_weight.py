from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
)


class Cosmos3TextPreAndPostLayerWeight(PreAndPostLayerWeight):
    """Load only the causal text embedding, final norm, and LM head."""

    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]

        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="embed_tokens.weight",
            data_type=self.data_type_,
        )
        tie_word_embeddings = network_config.get("tie_word_embeddings", False)
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=self.data_type_,
            embedding_weight=self.wte_weight_ if tie_word_embeddings else None,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="norm.weight",
            data_type=self.data_type_,
        )
