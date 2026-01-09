from lightllm.models.bloom.layer_infer.transformer_layer_infer import BloomTransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from functools import partial


class StarcoderTransformerLayerInfer(BloomTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self._bind_func()
        return

    def _bind_func(self):
        self._context_attention_kernel = partial(LlamaTransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_attention_kernel, self)
        return
