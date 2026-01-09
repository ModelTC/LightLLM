from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Gemma3PostLayerInfer(LlamaPostLayerInfer):
    """ """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = 1e-6
        return
