from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg


class Glm4MoeLiteMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg

        mtp_layer_idx = network_config["num_hidden_layers"]
        hidden_size = network_config["hidden_size"]

        self.eh_proj_weight_ = ROWMMWeight(
            in_dim=hidden_size * 2,
            out_dims=[hidden_size],
            weight_names=f"model.layers.{mtp_layer_idx}.eh_proj.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(mtp_layer_idx, "eh_proj"),
            tp_rank=0,
            tp_world_size=1,
        )

        self.enorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"model.layers.{mtp_layer_idx}.enorm.weight",
            data_type=self.data_type_,
        )

        self.hnorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"model.layers.{mtp_layer_idx}.hnorm.weight",
            data_type=self.data_type_,
        )

        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"model.layers.{mtp_layer_idx}.shared_head.norm.weight",
            data_type=self.data_type_,
        )

        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
