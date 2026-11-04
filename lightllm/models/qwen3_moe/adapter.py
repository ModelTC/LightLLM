from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.qwen3.adapter import Qwen3ModelAdapter


def qwen3_moe_specs(config, dtype, layer_ids):
    from lightllm.core.compute.moe.spec import MoeSpec

    experts = config.get("num_experts", 0)
    if not experts:
        return {}
    spec = MoeSpec(
        config["hidden_size"],
        config["moe_intermediate_size"],
        experts,
        config["num_experts_per_tok"],
        dtype,
        renormalize=config.get("norm_topk_prob", True),
    )
    return {
        layer: spec
        for layer in layer_ids
        if layer not in config.get("mlp_only_layers", []) and (layer + 1) % config.get("decoder_sparse_step", 1) == 0
    }


class Qwen3MoEModelAdapter(Qwen3ModelAdapter):
    transformer_weight_class = Qwen3MOETransformerLayerWeight
    transformer_layer_infer_class = Qwen3MOETransformerLayerInfer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.setdefault("norm_topk_prob", True)

    def moe_specs(self):
        return qwen3_moe_specs(self.config, self.dtype, self.layer_ids)
