import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, NoTpNormWeight, ROWMMWeight


class Internlm2RewardPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        self.wte_weight_ = EmbeddingWeight(
            weight_name="model.tok_embeddings.weight",
            data_type=self.data_type_,
        )
        self.score_head_ = ROWMMWeight(
            weight_names="v_head.weight",
            data_type=self.data_type_,
            name="score_head",
            tp_rank=0,
            tp_world_size=1,
        )
        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="model.norm.weight",
            data_type=self.data_type_,
        )
        return
