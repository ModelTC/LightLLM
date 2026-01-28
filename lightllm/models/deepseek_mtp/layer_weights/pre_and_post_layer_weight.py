from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg


class Deepseek3MTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg
        hidden_size = network_config["hidden_size"]
        self.eh_proj_weight_ = ROWMMWeight(
            in_dim=hidden_size * 2,
            out_dims=[hidden_size],
            weight_names="model.layers.0.eh_proj.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(0, "eh_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.layers.0.enorm.weight",
            data_type=self.data_type_,
        )
        self.hnorm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.layers.0.hnorm.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.layers.0.shared_head.norm.weight",
            data_type=self.data_type_,
        )

        # 与DeepseekV3模型共享, 不通过 load 加载
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        return
