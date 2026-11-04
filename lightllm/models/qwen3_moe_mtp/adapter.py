from copy import deepcopy
from lightllm.models.qwen3_moe.adapter import Qwen3MoEModelAdapter, qwen3_moe_specs
from lightllm.models.qwen3_moe_mtp.layer_weights.pre_and_post_layer_weight import Qwen3MOEMTPPreAndPostLayerWeight
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_infer.transformer_layer_infer import Qwen3MOEMTPTransformerLayerInfer
from lightllm.models.qwen3_moe_mtp.layer_weights.transformer_layer_weight import Qwen3MOEMTPTransformerLayerWeight
from lightllm.models.base.adapter import RecipeModelAdapter


class Qwen3MoEMTPModelAdapter(RecipeModelAdapter):
    pre_and_post_weight_class = Qwen3MOEMTPPreAndPostLayerWeight
    transformer_weight_class = Qwen3MOEMTPTransformerLayerWeight
    pre_layer_infer_class = Deepseek3MTPPreLayerInfer
    post_layer_infer_class = Qwen3MoEModelAdapter.post_layer_infer_class
    transformer_layer_infer_class = Qwen3MOEMTPTransformerLayerInfer

    def __init__(self, config, *, dtype, layer_ids, checkpoint_layer_ids, shared_weights):
        self.config = deepcopy(config)
        self.config.setdefault("norm_topk_prob", True)
        self.dtype = dtype
        self.layer_ids = tuple(layer_ids)
        self.checkpoint_layer_ids = tuple(checkpoint_layer_ids)
        depth = config["num_hidden_layers"]
        for ids in (self.layer_ids, self.checkpoint_layer_ids):
            if depth <= 0 or len(ids) != depth or len(set(ids)) != depth or min(ids) < 0:
                raise ValueError("draft layer IDs must uniquely describe every model layer")
        self.shared_weights = shared_weights
        self._attention = {}
        self._states = ()

    def moe_specs(self):
        specs = qwen3_moe_specs(self.config, self.dtype, self.checkpoint_layer_ids)
        return {
            layer: specs[checkpoint]
            for layer, checkpoint in zip(self.layer_ids, self.checkpoint_layer_ids, strict=True)
            if checkpoint in specs
        }

    def build_weights(self, quant_config):
        pre_post = self.pre_and_post_weight_class(self.dtype, network_config=self.config, quant_cfg=quant_config)
        pre_post.wte_weight_, pre_post.lm_head_weight_, pre_post.final_norm_weight_ = self.shared_weights
        layers = [
            self.transformer_weight_class(i, self.dtype, network_config=self.config, quant_cfg=quant_config)
            for i in self.checkpoint_layer_ids
        ]
        return pre_post, layers

    def build_recipe(self):
        pre = self.pre_layer_infer_class(network_config=self.config)
        post = self.post_layer_infer_class(network_config=self.config)
        layers = []
        for layer_id, checkpoint_id in zip(self.layer_ids, self.checkpoint_layer_ids, strict=True):
            # Dense/MoE selection belongs to checkpoint coordinates, not the shared execution domain.
            layer = self.transformer_layer_infer_class(checkpoint_id, network_config=self.config)
            layer.layer_num_ = layer_id
            layers.append(layer)
        return pre, post, layers
