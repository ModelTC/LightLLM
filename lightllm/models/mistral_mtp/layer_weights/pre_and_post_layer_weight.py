from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
    ROWMMWeight,
)


class MistralMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        self.eh_proj_weight_ = ROWMMWeight(
            weight_names="mtp.eh_proj.weight",
            data_type=self.data_type_,
            layer_num=0,
            name="eh_proj",
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="mtp.enorm.weight",
            data_type=self.data_type_,
        )
        self.hnorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="mtp.hnorm.weight",
            data_type=self.data_type_,
        )

        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        self.final_norm_weight_: RMSNormWeight = None
        return
