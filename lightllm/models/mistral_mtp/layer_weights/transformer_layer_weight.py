from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NoTpNormWeight


class MistralMTPTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight_names(self):
        self._gate_weight_name = f"mtp.layers.{self.layer_num_}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"mtp.layers.{self.layer_num_}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._down_weight_name = f"mtp.layers.{self.layer_num_}.mlp.down_proj.weight"
        self._down_bias_name = None

        self._ffn_norm_weight_name = f"mtp.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_weight(self):
        self._init_norm()
        self._init_ffn()

    def _init_ffn(self):
        self.gate_up_proj = ROWMMWeight(
            weight_names=[self._gate_weight_name, self._up_weight_name],
            data_type=self.data_type_,
            bias_names=[self._gate_bias_name, self._up_bias_name],
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="gate_up_proj",
        )
        self.down_proj = COLMMWeight(
            weight_names=self._down_weight_name,
            data_type=self.data_type_,
            bias_names=self._down_bias_name,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="down_proj",
        )

    def _init_norm(self):
        self.ffn_norm_weight_ = NoTpNormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )
