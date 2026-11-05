from lightllm.common.execution.contracts.model import ModelWeights
from lightllm.models.qwen3_5.adapter import Qwen35ModelAdapter
from lightllm.models.base.adapter import DenseModelAdapter
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import Qwen35TransformerLayerInfer
from lightllm.models.qwen3_5_mtp.layer_weights.pre_and_post_layer_weight import Qwen3_5MTPPreAndPostLayerWeight
from lightllm.models.qwen3_5_mtp.layer_weights.transformer_layer_weight import Qwen3_5MTPTransformerLayerWeight
from lightllm.models.qwen3_5_mtp.layer_infer.pre_layer_infer import Qwen3_5MTPPreLayerInfer


class Qwen35MTPModelAdapter(DenseModelAdapter):
    pre_and_post_weight_class = Qwen3_5MTPPreAndPostLayerWeight
    transformer_weight_class = Qwen3_5MTPTransformerLayerWeight
    pre_layer_infer_class = Qwen3_5MTPPreLayerInfer
    post_layer_infer_class = Qwen35ModelAdapter.post_layer_infer_class
    transformer_layer_infer_class = Qwen35TransformerLayerInfer

    def __init__(self, config, *, role, dtype, tp_world_size, layer_id, checkpoint_layer_id, shared_weights):
        super().__init__(config, role=role, dtype=dtype, tp_world_size=tp_world_size, layer_ids=(layer_id,))
        self.checkpoint_layer_id = checkpoint_layer_id
        self.shared_weights = shared_weights

    def build_weights(self, quant_config) -> ModelWeights:
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config, quant_cfg=quant_config)
        pre_post.wte_weight_, pre_post.lm_head_weight_, pre_post.main_norm_weight_ = self.shared_weights
        weight = self.transformer_weight_class(
            self.checkpoint_layer_id, self.dtype, network_config=self.config, quant_cfg=quant_config
        )
        return ModelWeights(pre_post, (weight,))
