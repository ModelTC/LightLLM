from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight


class StablelmTransformerLayerWeight(Qwen2TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self._att_norm_bias_name = f"model.layers.{self.layer_num_}.input_layernorm.bias"
        self._ffn_norm_bias_name = f"model.layers.{self.layer_num_}.post_attention_layernorm.bias"
