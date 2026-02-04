import os
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, FusedMoeWeight


class Qwen3OmniMOEThinkerTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        self.layer_num_ = network_config["num_hidden_layers"]
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return
