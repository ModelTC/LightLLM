from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class InternlmTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        self._o_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"
