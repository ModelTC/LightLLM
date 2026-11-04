from dataclasses import replace
from lightllm.models.qwen3_dflash.adapter import Qwen3DFlashModelAdapter
from lightllm.models.qwen3_dflash2.layer_infer.post_layer_infer import Qwen3DFlash2PostLayerInfer
from lightllm.models.qwen3_dflash2.layer_infer.transformer_layer_infer import Qwen3DFlash2TransformerLayerInfer
from lightllm.models.qwen3_dflash2.layer_weights.pre_and_post_layer_weight import Qwen3DFlash2PreAndPostLayerWeight
from lightllm.models.qwen3_dflash2.layer_weights.transformer_layer_weight import Qwen3DFlash2TransformerLayerWeight


class Qwen3DFlash2ModelAdapter(Qwen3DFlashModelAdapter):
    pre_and_post_weight_class = Qwen3DFlash2PreAndPostLayerWeight
    transformer_weight_class = Qwen3DFlash2TransformerLayerWeight
    post_layer_infer_class = Qwen3DFlash2PostLayerInfer
    transformer_layer_infer_class = Qwen3DFlash2TransformerLayerInfer
    share_main_weights = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = int(self.config.get("sliding_window", 0) or 0)
        window = (width - 1, width - 1) if width > 0 else (-1, -1)
        self._attention = {i: replace(spec, window=window) for i, spec in self._attention.items()}
