from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LayerNormWeight,
    NoTpPosEmbeddingWeight,
    LMHeadWeight,
)


class StarcoderPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)

    def _create_weight(self):
        hidden_size = self.network_config["hidden_size"]
        vocab_size = self.network_config["vocab_size"]
        max_position_embeddings = self.network_config["max_position_embeddings"]
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="transformer.wte.weight",
            data_type=self.data_type_,
        )
        self.wpe_weight_ = NoTpPosEmbeddingWeight(
            dim=hidden_size,
            max_position_embeddings=max_position_embeddings,
            weight_name="transformer.wpe.weight",
            data_type=self.data_type_,
        )

        self.final_norm_weight_ = LayerNormWeight(
            dim=hidden_size,
            weight_name="transformer.ln_f.weight",
            bias_name="transformer.ln_f.bias",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=self.data_type_,
        )
        return
