from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    NoTpNormWeight,
    NoTpPosEmbeddingWeight,
    LMHeadWeight,
)


class StarcoderPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)

        self.wte_weight_ = EmbeddingWeight(
            weight_name="transformer.wte.weight",
            data_type=self.data_type_,
        )
        self.wpe_weight_ = NoTpPosEmbeddingWeight(
            weight_name="transformer.wpe.weight",
            data_type=self.data_type_,
        )

        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="transformer.ln_f.weight",
            bias_name="transformer.ln_f.bias",
            data_type=self.data_type_,
        )
        self.lm_head_weight_ = LMHeadWeight(
            weight_name="lm_head.weight",
            data_type=self.data_type_,
        )
        return
