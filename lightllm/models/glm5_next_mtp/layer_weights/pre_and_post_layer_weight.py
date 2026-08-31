from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg
from lightllm.models.glm5_next.layer_weights.pre_and_post_layer_weight import (
    add_language_model_aliases,
)


class Glm5NextMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        layer_idx = network_config["num_hidden_layers"]
        hidden_size = network_config["hidden_size"]
        prefix = f"model.layers.{layer_idx}"
        self.eh_proj_weight_ = ROWMMWeight(
            in_dim=hidden_size * 2,
            out_dims=[hidden_size],
            weight_names=f"{prefix}.eh_proj.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"{prefix}.enorm.weight",
            data_type=self.data_type_,
        )
        self.hnorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"{prefix}.hnorm.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"{prefix}.shared_head.norm.weight",
            data_type=self.data_type_,
        )
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None

    def load_hf_weights(self, weights):
        add_language_model_aliases(weights)
        return super().load_hf_weights(weights)
