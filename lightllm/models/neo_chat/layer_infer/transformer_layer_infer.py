from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer


class NeoChatMOETransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        return
