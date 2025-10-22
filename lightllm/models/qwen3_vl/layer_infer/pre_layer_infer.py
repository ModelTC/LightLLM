from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Qwen3VLMultimodalPreLayerInfer(LlamaMultimodalPreLayerInfer):
    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.use_deepstack = True
        return
