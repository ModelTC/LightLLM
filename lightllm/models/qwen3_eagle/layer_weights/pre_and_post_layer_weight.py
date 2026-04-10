import numpy as np
import torch
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    ROWMMWeight,
    LMHeadWeight,
    RMSNormWeight,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class Qwen3EaglePreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        super().__init__(data_type, network_config)
        self.quant_cfg: Quantcfg = quant_cfg
        hidden_size = network_config["hidden_size"]
        self.final_norm_weight_: RMSNormWeight = RMSNormWeight(
            dim=hidden_size,
            weight_name="norm.weight",
            data_type=self.data_type_,
        )
        
        try:
            self.d2t_weight_: RMSNormWeight = RMSNormWeight(
                dim=network_config["draft_vocab_size"],
                weight_name="d2t",
                data_type=torch.int64
            )
            
            self.t2d_weight_: RMSNormWeight = RMSNormWeight(
                dim=network_config["target_vocab_size"],
                weight_name="t2d",
                data_type=torch.bool
            )
        except Exception as e:
            self.d2t_weight_ = None
            self.t2d_weight_ = None
            logger.warning(f"No d2t and t2d weight found in Eagle3 Model.")
        
        try:
            self.lm_head_weight_: LMHeadWeight = LMHeadWeight(
                dim=hidden_size,
                vocab_size=network_config["draft_vocab_size"],
                weight_name="lm_head.weight",
                data_type=self.data_type_,
            )
        except Exception as e:
            self.lm_head_weight_ = None
            logger.warning(f"Failed to initialize lm_head_weight_, error: {e}")
        # 与Qwen3模型共享
        self.wte_weight_: EmbeddingWeight = None
        
        return