from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class StarcoderTransformerLayerInfer(BloomTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self._bind_func()
        return

    def _bind_func(self):
        LlamaTransformerLayerInfer._bind_attention(self)
        return
