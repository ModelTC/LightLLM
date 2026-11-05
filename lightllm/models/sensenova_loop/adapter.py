from lightllm.core.contracts.model import ModelWeights
from lightllm.models.qwen3_5.adapter import Qwen35ModelAdapter
from lightllm.models.sensenova_loop.layer_infer.transformer_layer_infer import SenseNovaLoopTransformerLayerInfer
from lightllm.models.sensenova_loop.layer_weights.transformer_layer_weight import SenseNovaLoopTransformerLayerWeight


class SenseNovaLoopModelAdapter(Qwen35ModelAdapter):
    transformer_weight_class = SenseNovaLoopTransformerLayerWeight
    transformer_layer_infer_class = SenseNovaLoopTransformerLayerInfer

    def build_weights(self, quant_config) -> ModelWeights:
        indices = self.config["loop_physical_indices"]
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config)
        physical = {
            i: self.transformer_weight_class(i, self.dtype, network_config=self.config, quant_cfg=quant_config)
            for i in dict.fromkeys(indices)
        }
        return ModelWeights(pre_post, tuple(physical[i] for i in indices))
