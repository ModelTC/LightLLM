from dataclasses import replace
from lightllm.models.qwen3_5.adapter import Qwen35ModelAdapter
from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import Qwen35MOETransformerLayerWeight


class Qwen35MoEModelAdapter(Qwen35ModelAdapter):
    transformer_weight_class = Qwen35MOETransformerLayerWeight

    def moe_specs(self):
        return {
            layer: replace(spec, round_intermediates=True) for layer, spec in Qwen35ModelAdapter.moe_specs(self).items()
        }
