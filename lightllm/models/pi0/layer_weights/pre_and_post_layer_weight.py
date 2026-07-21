from __future__ import annotations

from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    NoTpGEMMANormWeight,
    ROWMMWeight,
)
from lightllm.common.basemodel.layer_weights.pre_and_post_layer_weight import (
    PreAndPostLayerWeight,
)
from lightllm.common.quantization import Quantcfg


class Pi0VLMPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        embedding_name = "paligemma_with_expert.paligemma.lm_head.weight"
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name=embedding_name,
            data_type=data_type,
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name=embedding_name,
            data_type=data_type,
            embedding_weight=self.wte_weight_,
        )
        self.final_norm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name=("paligemma_with_expert.paligemma.model.language_model.norm.weight"),
            data_type=data_type,
        )


class Pi0ActionPreAndPostLayerWeight(PreAndPostLayerWeight):
    """Replicated action projections backed by LightLLM meta weights."""

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        self.quant_cfg = quant_cfg
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        max_action_dim = network_config["max_action_dim"]
        max_state_dim = network_config["max_state_dim"]
        is_pi05 = bool(network_config["is_pi05"])

        def row(in_dim, out_dim, weight_name, bias_name, quant_name):
            return ROWMMWeight(
                in_dim=in_dim,
                out_dims=[out_dim],
                weight_names=weight_name,
                data_type=data_type,
                bias_names=bias_name,
                quant_method=quant_cfg.get_quant_method(-1, quant_name),
                tp_rank=0,
                tp_world_size=1,
            )

        self.action_in_proj = row(
            max_action_dim,
            hidden_size,
            "action_in_proj.weight",
            "action_in_proj.bias",
            "action_in_proj",
        )
        self.action_out_proj = row(
            hidden_size,
            max_action_dim,
            "action_out_proj.weight",
            "action_out_proj.bias",
            "action_out_proj",
        )
        if is_pi05:
            time_prefix = "time_mlp"
            time_input_size = hidden_size
            self.state_proj = None
            self.final_norm_weight_ = None
            self.final_norm_dense = row(
                hidden_size,
                3 * hidden_size,
                "paligemma_with_expert.gemma_expert.model.norm.dense.weight",
                "paligemma_with_expert.gemma_expert.model.norm.dense.bias",
                "final_norm_dense",
            )
        else:
            time_prefix = "action_time_mlp"
            time_input_size = 2 * hidden_size
            self.state_proj = row(
                max_state_dim,
                hidden_size,
                "state_proj.weight",
                "state_proj.bias",
                "state_proj",
            )
            self.final_norm_weight_ = NoTpGEMMANormWeight(
                dim=hidden_size,
                weight_name="paligemma_with_expert.gemma_expert.model.norm.weight",
                data_type=data_type,
            )
            self.final_norm_dense = None

        self.time_mlp_in = row(
            time_input_size,
            hidden_size,
            f"{time_prefix}_in.weight",
            f"{time_prefix}_in.bias",
            "time_mlp_in",
        )
        self.time_mlp_out = row(
            hidden_size,
            hidden_size,
            f"{time_prefix}_out.weight",
            f"{time_prefix}_out.bias",
            "time_mlp_out",
        )
