import torch
from typing import final
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.qwen3.model import Qwen3TpPartModel, Qwen3ModelAdapter
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager


logger = init_logger(__name__)


def qwen3_moe_specs(config, dtype, layer_ids):
    from lightllm.common.execution.specs import MoeSpec

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


class Qwen3MOEModel(Qwen3TpPartModel):
    model_adapter_class = Qwen3MoEModelAdapter
    # weight class
    transformer_weight_class = Qwen3MOETransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3MOETransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    @classmethod
    def normalize_model_config(cls, model_config):
        config = super().normalize_model_config(model_config)
        if "num_experts" not in config and "num_local_experts" in config:
            config["num_experts"] = config["num_local_experts"]
        return config

    def _init_custom(self):
        super()._init_custom()
        # Only initialize DeepEP group for MoE models with num_experts
        if "num_experts" in self.config and self.config["num_experts"] > 0:
            dist_group_manager.new_deepep_group(
                n_routed_experts=self.config["num_experts"],
                hidden_size=self.config["hidden_size"],
                expert_quant_method_names=dist_group_manager.get_moe_quant_methods(self.trans_layers_weight),
                num_experts_per_tok=self.config.get("num_experts_per_tok", 1),
                moe_intermediate_size=self.config.get("moe_intermediate_size", self.config.get("intermediate_size")),
            )
