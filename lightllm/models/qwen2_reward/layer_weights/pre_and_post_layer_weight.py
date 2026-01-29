import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight


class Qwen2RewardPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        del self.lm_head_weight_
        self.score_up_weight_ = ROWMMWeight(
            weight_names="score.0.weight",
            bias_names="score.0.bias",
            data_type=self.data_type_,
            name="score_up_weight",
            tp_rank=0,
            tp_world_size=1,
        )
        self.score_down_weight_ = ROWMMWeight(
            weight_names="score.2.weight",
            bias_names="score.2.bias",
            data_type=self.data_type_,
            name="score_down_weight",
            tp_rank=0,
            tp_world_size=1,
        )
        return
