from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, NoTpNormWeight


class ChatGLM2PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)

        self.wte_weight_ = EmbeddingWeight(
            weight_name="transformer.embedding.word_embeddings.weight", data_type=self.data_type_
        )
        self.lm_head_weight_ = LMHeadWeight(
            weight_name="transformer.output_layer.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="transformer.encoder.final_layernorm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
