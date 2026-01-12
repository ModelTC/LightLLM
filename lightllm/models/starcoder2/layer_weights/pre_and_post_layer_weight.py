from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, LayerNormWeight


class Starcoder2PreAndPostLayerWeight(PreAndPostLayerWeight):
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
        tie_word_embeddings = self.network_config_.get("tie_word_embeddings", False)
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight",
            data_type=self.data_type_,
        )

        self.final_norm_weight_ = LayerNormWeight(
            dim=hidden_size,
            weight_name="model.norm.weight",
            data_type=self.data_type_,
            bias_name="model.norm.bias",
        )
        return
