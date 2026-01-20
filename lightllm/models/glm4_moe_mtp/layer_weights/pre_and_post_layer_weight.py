from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    ROWMMWeight,
    LMHeadWeight,
    NoTpNormWeight,
)


class Glm4MoeMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    """
    Pre and post layer weights for GLM-4.7 MoE MTP model.

    The MTP model has its own projection and normalization weights,
    but shares the embedding, lm_head, and final_norm with the main model.
    """

    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)

        # MTP-specific projection weight
        self.eh_proj_weight_ = ROWMMWeight(
            weight_names="model.layers.0.proj.weight",
            data_type=self.data_type_,
            name="eh_proj",
            tp_rank=0,
            tp_world_size=1,
        )

        # MTP-specific normalization weights
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

        # Shared with main GLM4MOE model (set during model init)
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        self.final_norm_weight_: NoTpNormWeight = None
        return
