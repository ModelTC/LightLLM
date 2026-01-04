from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    NoTpNormWeight,
    ROWMMWeight,
)


class MistralMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)

        self.eh_proj_weight_ = ROWMMWeight(
            weight_names="mtp.eh_proj.weight",
            data_type=self.data_type_,
            layer_num=0,
            name="eh_proj",
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = NoTpNormWeight(
            weight_name="mtp.enorm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        self.hnorm_weight_ = NoTpNormWeight(
            weight_name="mtp.hnorm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )

        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        self.final_norm_weight_: NoTpNormWeight = None
        return
