from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
from lightllm.common.quantization import Quantcfg
from lightllm.models.pi0.layer_weights.pre_and_post_layer_weight import (
    Pi0ActionPreAndPostLayerWeight,
)


class Pi05ActionPreAndPostLayerWeight(Pi0ActionPreAndPostLayerWeight):
    """Pi0.5 projections for timestep-conditioned action tokens."""

    time_mlp_prefix = "time_mlp"
    time_mlp_input_multiplier = 1

    def _init_variant_weights(
        self,
        hidden_size: int,
        data_type,
        network_config: dict,
        quant_cfg: Quantcfg,
    ) -> None:
        self.final_norm_dense = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[3 * hidden_size],
            weight_names="paligemma_with_expert.gemma_expert.model.norm.dense.weight",
            data_type=data_type,
            bias_names="paligemma_with_expert.gemma_expert.model.norm.dense.bias",
            quant_method=quant_cfg.get_quant_method(-1, "final_norm_dense"),
            tp_rank=0,
            tp_world_size=1,
        )
