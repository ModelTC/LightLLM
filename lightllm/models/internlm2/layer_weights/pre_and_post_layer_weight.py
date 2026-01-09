from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, NoTpNormWeight


class Internlm2PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        self.wte_weight_ = EmbeddingWeight(weight_name="model.tok_embeddings.weight", data_type=self.data_type_)
        self.lm_head_weight_ = LMHeadWeight(weight_name="output.weight", data_type=self.data_type_)

        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="model.norm.weight",
            data_type=self.data_type_,
        )
        return
