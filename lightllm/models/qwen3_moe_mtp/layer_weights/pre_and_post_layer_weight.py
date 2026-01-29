import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    ROWMMWeight,
    LMHeadWeight,
    RMSNormWeight,
)
from lightllm.common.quantization import Quantcfg


class Qwen3MOEMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg
        hidden_size = network_config["hidden_size"]
        self.eh_proj_weight_ = ROWMMWeight(
            in_dim=hidden_size * 2,
            out_dims=[hidden_size],
            weight_names="model.layers.0.proj.weight",
            quant_method=self.quant_cfg.get_quant_method(0, "eh_proj"),
            data_type=self.data_type_,
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.layers.0.norm_after_embedding.weight",
            data_type=self.data_type_,
        )
        self.hnorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.layers.0.norm_before_output.weight",
            data_type=self.data_type_,
        )
        # 与Qwen3MOE模型共享
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        self.final_norm_weight_: RMSNormWeight = None
        return
