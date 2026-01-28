from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LayerNormWeight, LMHeadWeight


class BloomPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        self.pre_norm_weight_ = LayerNormWeight(
            dim=hidden_size,
            weight_name="word_embeddings_layernorm.weight",
            data_type=self.data_type_,
            bias_name="word_embeddings_layernorm.bias",
        )
        self.final_norm_weight_ = LayerNormWeight(
            dim=hidden_size,
            weight_name="ln_f.weight",
            data_type=self.data_type_,
            bias_name="ln_f.bias",
        )

        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="word_embeddings.weight",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="word_embeddings.weight",
            data_type=self.data_type_,
            embedding_weight=self.wte_weight_,
        )
