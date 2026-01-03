from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    NoTpNormWeight,
    ROWMMWeight,
)


class Deepseek3MTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)

        self.eh_proj_weight_ = ROWMMWeight(
            weight_names="model.layers.0.eh_proj.weight",
            data_type=self.data_type_,
            name="eh_proj",
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = NoTpNormWeight(
            weight_name="model.layers.0.enorm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        self.hnorm_weight_ = NoTpNormWeight(
            weight_name="model.layers.0.hnorm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="model.layers.0.shared_head.norm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )

        # 与DeepseekV3模型共享, 不通过 load 加载
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        return
