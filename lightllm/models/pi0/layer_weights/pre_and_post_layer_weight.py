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
            weight_name=(
                "paligemma_with_expert.paligemma.model.language_model.norm.weight"
            ),
            data_type=data_type,
        )


class Pi0ActionPreAndPostLayerWeight(PreAndPostLayerWeight):
    """Pi0 projections for the continuous-state action suffix."""

    time_mlp_prefix = "action_time_mlp"
    time_mlp_input_multiplier = 2

    def __init__(self, data_type, network_config, quant_cfg: Quantcfg):
        self.quant_cfg = quant_cfg
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        max_action_dim = network_config["max_action_dim"]

        self.action_in_proj = ROWMMWeight(
            in_dim=max_action_dim,
            out_dims=[hidden_size],
            weight_names="action_in_proj.weight",
            data_type=data_type,
            bias_names="action_in_proj.bias",
            quant_method=quant_cfg.get_quant_method(-1, "action_in_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.action_out_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[max_action_dim],
            weight_names="action_out_proj.weight",
            data_type=data_type,
            bias_names="action_out_proj.bias",
            quant_method=quant_cfg.get_quant_method(-1, "action_out_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self._init_variant_weights(hidden_size, data_type, network_config, quant_cfg)
        self.time_mlp_in = ROWMMWeight(
            in_dim=self.time_mlp_input_multiplier * hidden_size,
            out_dims=[hidden_size],
            weight_names=f"{self.time_mlp_prefix}_in.weight",
            data_type=data_type,
            bias_names=f"{self.time_mlp_prefix}_in.bias",
            quant_method=quant_cfg.get_quant_method(-1, "time_mlp_in"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.time_mlp_out = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[hidden_size],
            weight_names=f"{self.time_mlp_prefix}_out.weight",
            data_type=data_type,
            bias_names=f"{self.time_mlp_prefix}_out.bias",
            quant_method=quant_cfg.get_quant_method(-1, "time_mlp_out"),
            tp_rank=0,
            tp_world_size=1,
        )

    def _init_variant_weights(
        self,
        hidden_size: int,
        data_type,
        network_config: dict,
        quant_cfg: Quantcfg,
    ) -> None:
        self.state_proj = ROWMMWeight(
            in_dim=network_config["max_state_dim"],
            out_dims=[hidden_size],
            weight_names="state_proj.weight",
            data_type=data_type,
            bias_names="state_proj.bias",
            quant_method=quant_cfg.get_quant_method(-1, "state_proj"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.final_norm_weight_ = NoTpGEMMANormWeight(
            dim=hidden_size,
            weight_name="paligemma_with_expert.gemma_expert.model.norm.weight",
            data_type=data_type,
        )
