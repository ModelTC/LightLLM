from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import Qwen3NextTransformerLayerWeight
from lightllm.models.qwen3next.layer_weights.pre_and_post_layer_weight import Qwen3NextPreAndPostLayerWeight
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer
from lightllm.models.base.adapter import GDNModelAdapter
from lightllm.models.llama.adapter import LlamaModelAdapter


class Qwen3NextModelAdapter(GDNModelAdapter):
    pre_and_post_weight_class = Qwen3NextPreAndPostLayerWeight
    transformer_weight_class = Qwen3NextTransformerLayerWeight
    pre_layer_infer_class = LlamaModelAdapter.pre_layer_infer_class
    post_layer_infer_class = LlamaModelAdapter.post_layer_infer_class
    transformer_layer_infer_class = Qwen3NextTransformerLayerInfer

    def moe_specs(self):
        from lightllm.common.execution.compute.moe.spec import MoeSpec

        config = self.config
        experts = config.get("num_experts", 0)
        if not experts:
            return {}
        spec = MoeSpec(
            config["hidden_size"],
            config["moe_intermediate_size"],
            experts,
            config["num_experts_per_tok"],
            self.dtype,
            renormalize=config.get("norm_topk_prob", False),
            shared_experts=1,
            gated_shared=True,
        )
        return {
            layer: spec
            for layer in self.layer_ids
            if layer not in config.get("mlp_only_layers", [])
            and (layer + 1) % config.get("decoder_sparse_step", 1) == 0
        }
