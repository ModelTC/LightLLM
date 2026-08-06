from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer


class DeepseekV4MTPTransformerLayerInfer(DeepseekV4TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.is_last_layer = True
        return
