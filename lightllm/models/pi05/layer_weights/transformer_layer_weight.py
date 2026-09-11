from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
from lightllm.models.pi0.layer_weights.transformer_layer_weight import (
    Pi0ActionTransformerLayerWeight,
)


class Pi05ActionTransformerLayerWeight(Pi0ActionTransformerLayerWeight):
    """Pi0.5 expert weights with adaptive RMSNorm modulation."""

    def _init_weight_names(self):
        super()._init_weight_names()
        prefix = "paligemma_with_expert.gemma_expert.model.layers." f"{self.layer_num_}"
        self._att_norm_dense_weight_name = f"{prefix}.input_layernorm.dense.weight"
        self._att_norm_dense_bias_name = f"{prefix}.input_layernorm.dense.bias"
        self._ffn_norm_dense_weight_name = (
            f"{prefix}.post_attention_layernorm.dense.weight"
        )
        self._ffn_norm_dense_bias_name = f"{prefix}.post_attention_layernorm.dense.bias"

    def _init_norm(self):
        self.att_norm_dense = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[3 * self.n_embed],
            weight_names=self._att_norm_dense_weight_name,
            data_type=self.data_type_,
            bias_names=self._att_norm_dense_bias_name,
            quant_method=self.get_quant_method("att_norm_dense"),
            tp_rank=0,
            tp_world_size=1,
        )
        self.ffn_norm_dense = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[3 * self.n_embed],
            weight_names=self._ffn_norm_dense_weight_name,
            data_type=self.data_type_,
            bias_names=self._ffn_norm_dense_bias_name,
            quant_method=self.get_quant_method("ffn_norm_dense"),
            tp_rank=0,
            tp_world_size=1,
        )
