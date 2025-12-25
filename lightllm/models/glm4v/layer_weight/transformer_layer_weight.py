from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight
from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight


class Glm4VTransformerLayerWeight(Qwen2TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def _init_weight_names(self):
        self._post_self_att_norm_weight_name = f"model.layers.{self.layer_num_}.post_self_attn_layernorm.weight"
        self._post_self_att_norm_bias_name = None
        self._post_mlp_norm_weight_name = f"model.layers.{self.layer_num_}.post_mlp_layernorm.weight"
        self._post_mlp_norm_bias_name = None
        super()._init_weight_names()

    def load_hf_weights(self, weights):
        gate_up_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"
        if gate_up_weight_name in weights:
            intermediate_size = self.network_config_["intermediate_size"]
            gate_up_proj = weights[gate_up_weight_name]
            gate_weight_ = gate_up_proj[0:intermediate_size, :]
            up_weight_ = gate_up_proj[intermediate_size:, :]
            weights[self._gate_weight_name] = gate_weight_
            weights[self._up_weight_name] = up_weight_
            del weights[gate_up_weight_name]
        super().load_hf_weights(weights)

    def _init_norm(self):
        self._post_self_att_norm_weight_ = NormWeight(
            self._post_self_att_norm_weight_name, self.data_type_, bias_name=self._post_self_att_norm_bias_name
        )
        self._post_mlp_norm_weight_ = NormWeight(
            self._post_mlp_norm_weight_name, self.data_type_, bias_name=self._post_mlp_norm_bias_name
        )
        super()._init_norm()
