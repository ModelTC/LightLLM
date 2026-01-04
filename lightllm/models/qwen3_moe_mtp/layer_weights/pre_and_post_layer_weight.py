import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    ROWMMWeight,
    LMHeadWeight,
    NoTpNormWeight,
)


class Qwen3MOEMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)

        self.eh_proj_weight_ = ROWMMWeight(
            weight_names="model.layers.0.proj.weight",
            data_type=self.data_type_,
            name="eh_proj",
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = NoTpNormWeight(
            weight_name="model.layers.0.norm_after_embedding.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        self.hnorm_weight_ = NoTpNormWeight(
            weight_name="model.layers.0.norm_before_output.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        # 与Qwen3MOE模型共享
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        self.final_norm_weight_: NoTpNormWeight = None
        return
