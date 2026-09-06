from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight


class DeepseekV4MTPTransformerLayerWeight(DeepseekV4TransformerLayerWeight):
    def _parse_config(self):
        super()._parse_config()
        self.prefix = "mtp.0"
        return
