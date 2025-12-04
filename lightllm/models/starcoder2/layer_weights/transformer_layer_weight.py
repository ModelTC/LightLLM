from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Starcoder2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _parse_config(self):
        super()._parse_config()
        self.network_config_["intermediate_size"] = self.network_config_["hidden_size"] * 4

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        self._o_bias_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.bias"

        self._up_weight_name = f"model.layers.{self.layer_num_}.mlp.c_fc.weight"
        self._up_bias_name = f"model.layers.{self.layer_num_}.mlp.c_fc.bias"
        self._down_weight_name = f"model.layers.{self.layer_num_}.mlp.c_proj.weight"
        self._down_bias_name = f"model.layers.{self.layer_num_}.mlp.c_proj.bias"

        self._att_norm_bias_name = f"model.layers.{self.layer_num_}.input_layernorm.bias"
        self._ffn_norm_bias_name = f"model.layers.{self.layer_num_}.post_attention_layernorm.bias"

    def _init_ffn(self):
        self.up_proj = ROWMMWeight(
            weight_names=self._up_weight_name,
            data_type=self.data_type_,
            bias_names=self._up_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="up_proj",
        )
        self.down_proj = COLMMWeight(
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=self._down_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="down_proj",
        )
