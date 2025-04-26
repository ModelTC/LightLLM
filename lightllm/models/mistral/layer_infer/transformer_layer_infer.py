from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class MistralTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.head_dim_ = network_config.get("head_dim", self.head_dim_)
        return
