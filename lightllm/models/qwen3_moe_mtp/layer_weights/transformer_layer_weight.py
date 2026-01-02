import os
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpNormWeight


class Qwen3MOEMTPTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight(self):
        self._init_norm()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()

    def _init_norm(self):
        self.ffn_norm_weight_ = NoTpNormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )
