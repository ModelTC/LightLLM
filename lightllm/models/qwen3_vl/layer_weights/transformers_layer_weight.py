from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight


class Qwen3VLTransformerLayerWeight(Qwen3TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._q_bias_name = None
        self._k_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_norm_name = f"model.language_model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._k_bias_name = None
        self._v_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._o_weight_name = f"model.language_model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._att_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"model.language_model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

        self._gate_weight_name = f"model.language_model.layers.{self.layer_num_}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"model.language_model.layers.{self.layer_num_}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._down_weight_name = f"model.language_model.layers.{self.layer_num_}.mlp.down_proj.weight"
        self._down_bias_name = None
