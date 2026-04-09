import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    ROWMMWeight,
    LMHeadWeight,
    RMSNormWeight,
)
from lightllm.common.quantization import Quantcfg

"""Qwen3-eagle Weights List
d2t	[32000]	
fc.weight	[2048, 6144]		
midlayer.hidden_norm.weight	[2048]	
midlayer.input_layernorm.weight	[2048]	
midlayer.mlp.down_proj.weight	[2048, 6144]	
midlayer.mlp.gate_proj.weight	[6144, 2048]	
midlayer.mlp.up_proj.weight	[6144, 2048]	
midlayer.post_attention_layernorm.weight	[2048]	
midlayer.self_attn.k_proj.weight	[512, 4096]	
midlayer.self_attn.o_proj.weight	[2048, 4096]	
midlayer.self_attn.q_proj.weight	[4096, 4096]	
midlayer.self_attn.v_proj.weight	[512, 4096]	
norm.weight	[2048]	
t2d	[151936]	
lm_head.weight	[32000, 2048]	
"""

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
            self.lm_head_weight_: LMHeadWeight = LMHeadWeight(
                dim=hidden_size,
                vocab_size=network_config["draft_vocab_size"],
                weight_name="lm_head.weight",
                data_type=self.data_type_,
            )
        except Exception as e:
            self.lm_head_weight_ = None
        
        # 与Qwen3模型共享
        self.wte_weight_: EmbeddingWeight = None
        
        return