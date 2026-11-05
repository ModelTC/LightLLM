from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    NoTpGEMMANormWeight,
    ROWMMWeight,
)
from lightllm.common.quantization import Quantcfg


class Qwen3_5MTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg
        hidden_size = network_config["hidden_size"]

        self.eh_proj_weight_ = ROWMMWeight(
            in_dim=hidden_size * 2,
            out_dims=[hidden_size],
            weight_names="mtp.fc.weight",
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(0, "eh_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="mtp.pre_fc_norm_embedding.weight",
            data_type=self.data_type_,
        )
        self.hnorm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="mtp.pre_fc_norm_hidden.weight",
            data_type=self.data_type_,
        )
        self.final_norm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="mtp.norm.weight",
            data_type=self.data_type_,
        )

        # Shared with the main Qwen3.5 model, injected by the model class (not loaded here).
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        return
