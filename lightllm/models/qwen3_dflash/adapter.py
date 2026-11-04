from dataclasses import replace
from lightllm.models.llama.adapter import LlamaModelAdapter
from lightllm.models.qwen3_dflash.layer_infer.post_layer_infer import Qwen3DFlashPostLayerInfer
from lightllm.models.qwen3_dflash.layer_infer.pre_layer_infer import Qwen3DFlashPreLayerInfer
from lightllm.models.qwen3_dflash.layer_infer.transformer_layer_infer import Qwen3DFlashTransformerLayerInfer
from lightllm.models.qwen3_dflash.layer_weights.pre_and_post_layer_weight import Qwen3DFlashPreAndPostLayerWeight
from lightllm.models.qwen3_dflash.layer_weights.transformer_layer_weight import Qwen3DFlashTransformerLayerWeight


class Qwen3DFlashModelAdapter(LlamaModelAdapter):
    attention_phases = ("decode",)
    pre_and_post_weight_class = Qwen3DFlashPreAndPostLayerWeight
    transformer_weight_class = Qwen3DFlashTransformerLayerWeight
    pre_layer_infer_class = Qwen3DFlashPreLayerInfer
    post_layer_infer_class = Qwen3DFlashPostLayerInfer
    transformer_layer_infer_class = Qwen3DFlashTransformerLayerInfer
    share_main_weights = False

    def __init__(self, *args, main_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self._attention = {i: replace(spec, causal=False) for i, spec in self._attention.items()}
        self.main_weights = main_weights

    def build_weights(self, quant_config):
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config, quant_cfg=quant_config)
        if self.share_main_weights:
            pre_post.wte_weight_ = self.main_weights.wte_weight_
            pre_post.lm_head_weight_ = self.main_weights.lm_head_weight_
        layers = [
            self.transformer_weight_class(i, self.dtype, network_config=self.config, quant_cfg=quant_config)
            for i in range(len(self.layer_ids))
        ]
        return pre_post, layers
