from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NormWeight


class MistralMTPTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)

        self._gate_weight_name = f"mtp.layers.{self.layer_num_}.mlp.gate_proj.weight"
        self._up_weight_name = f"mtp.layers.{self.layer_num_}.mlp.up_proj.weight"
        self._down_weight_name = f"mtp.layers.{self.layer_num_}.mlp.down_proj.weight"

        self._ffn_norm_weight_name = f"mtp.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None
        return

    def _init_weight(self):
        self._init_norm()
        self._init_ffn()

    def _init_norm(self):
        self.ffn_norm_weight_ = NormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )
