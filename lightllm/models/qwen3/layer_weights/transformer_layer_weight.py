from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    NormWeight,
)


class Qwen3TransformerLayerWeight(Qwen2TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_norm_name = f"model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._k_norm_name = f"model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._q_bias_name = None
        self._k_bias_name = None
        self._v_bias_name = None

    def _init_norm(self):
        super()._init_norm()

        self.q_norm_weight_ = NormWeight(weight_name=self._q_norm_name, data_type=self.data_type_)
        self.k_norm_weight_ = NormWeight(weight_name=self._k_norm_name, data_type=self.data_type_)
