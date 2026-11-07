from lightllm.common.execution.contracts.model import ModelWeights
from lightllm.models.llama.adapter import LlamaModelAdapter
from lightllm.models.qwen3_eagle.layer_infer.pre_layer_infer import Qwen3EaglePreLayerInfer
from lightllm.models.qwen3_eagle.layer_infer.transformer_layer_infer import Qwen3EagleTransformerLayerInfer
from lightllm.models.qwen3_eagle.layer_weights.pre_and_post_layer_weight import Qwen3EaglePreAndPostLayerWeight
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight


class Qwen3EagleModelAdapter(LlamaModelAdapter):
    pre_and_post_weight_class = Qwen3EaglePreAndPostLayerWeight
    pre_layer_infer_class = Qwen3EaglePreLayerInfer
    transformer_weight_class = Qwen3EagleTransformerLayerWeight
    transformer_layer_infer_class = Qwen3EagleTransformerLayerInfer

    def __init__(self, *args, shared_embedding, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_embedding = shared_embedding

    def build_weights(self, quant_config) -> ModelWeights:
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config, quant_cfg=quant_config)
        if self.config.get("draft_vocab_size") is not None:
            pre_post.wte_weight_ = self.shared_embedding
        layers = [
            self.transformer_weight_class(i, self.dtype, network_config=self.config, quant_cfg=quant_config)
            for i in range(len(self.layer_ids))
        ]
        return ModelWeights(pre_post, tuple(layers))
