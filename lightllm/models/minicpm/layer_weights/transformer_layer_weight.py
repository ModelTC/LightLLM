import math
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class MiniCPMTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _parse_config(self):
        super()._parse_config()
        num_hidden_layers = self.network_config_["num_hidden_layers"]
        scale_depth = self.network_config_.get("scale_depth", math.sqrt(num_hidden_layers))
        self.layer_scale = scale_depth / math.sqrt(num_hidden_layers)

    def load_hf_weights(self, weights):
        if self._o_weight_name in weights:
            weights[self._o_weight_name] *= self.layer_scale
        if self._down_weight_name in weights:
            weights[self._down_weight_name] *= self.layer_scale
        super().load_hf_weights(weights)
