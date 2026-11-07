from lightllm.common.execution.contracts.model import ModelWeights
from lightllm.common.execution.contracts.model import ModelRecipe
from lightllm.models.deepseek2.adapter import Deepseek2ModelAdapter
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight
from lightllm.models.base.adapter import grouped_moe_specs


class Deepseek3MTPModelAdapter(Deepseek2ModelAdapter):
    moe_group_score_topk = 2
    pre_and_post_weight_class = Deepseek3MTPPreAndPostLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer

    def __init__(self, *args, checkpoint_layer_ids, shared_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_layer_ids = tuple(checkpoint_layer_ids)
        ids = self.checkpoint_layer_ids
        if len(ids) != len(self.layer_ids) or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("MLA checkpoint layer IDs must uniquely describe every model layer")
        self.shared_weights = shared_weights

    def moe_specs(self):
        specs = grouped_moe_specs(
            self.config, self.dtype, layer_ids=self.checkpoint_layer_ids, group_score_topk=self.moe_group_score_topk
        )
        return {
            layer: specs[checkpoint]
            for layer, checkpoint in zip(self.layer_ids, self.checkpoint_layer_ids, strict=True)
            if checkpoint in specs
        }

    def build_weights(self, quant_config) -> ModelWeights:
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config, quant_cfg=quant_config)
        pre_post.wte_weight_, pre_post.lm_head_weight_ = self.shared_weights
        layers = [
            self.transformer_weight_class(i, self.dtype, network_config=self.config, quant_cfg=quant_config)
            for i in self.checkpoint_layer_ids
        ]
        return ModelWeights(pre_post, tuple(layers))

    def build_recipe(self) -> ModelRecipe:
        pre = self.pre_layer_infer_class(network_config=self.config)
        post = self.post_layer_infer_class(network_config=self.config)
        layers = []
        for layer_id, checkpoint_id in zip(self.layer_ids, self.checkpoint_layer_ids, strict=True):
            # The checkpoint selects dense/MoE semantics; the execution domain selects state and routing slots.
            layer = self.transformer_layer_infer_class(checkpoint_id, network_config=self.config)
            layer.layer_num_ = layer_id
            self.validate_layer(layer)
            layers.append(layer)
        return ModelRecipe(pre, post, tuple(layers))
