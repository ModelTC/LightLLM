"""
Transformer layer weight classes for Qwen3.5 models.

This module extends Qwen3Next layer weight classes with Qwen3.5-specific
weight preprocessing, including:
- Fused MoE expert weight splitting (gate_up_proj -> gate_proj + up_proj)
- Linear-attention projection split (in_proj_qkv -> in_proj_q + in_proj_k + in_proj_v)
"""

import torch

from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextFullAttentionTransformerLayerWeight,
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def split_fused_expert_weights(weights, layer_num, moe_intermediate_size):
    """Split fused gate_up_proj weights into separate gate_proj and up_proj per expert.

    Qwen3.5-MoE stores expert weights in fused format:
    - gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
    - down_proj: [num_experts, hidden_size, moe_intermediate_size]

    The inherited FusedMoeWeight expects per-expert weights:
    - model.layers.X.mlp.experts.{expert_idx}.gate_proj.weight
    - model.layers.X.mlp.experts.{expert_idx}.up_proj.weight
    - model.layers.X.mlp.experts.{expert_idx}.down_proj.weight

    Args:
        weights: Dictionary mapping weight names to tensors. Modified in-place.
        layer_num: Layer number to process (only processes weights for this layer).
        moe_intermediate_size: Intermediate size for each MoE expert.
    """
    layer_prefix = f"model.layers.{layer_num}."
    keys = list(weights.keys())
    gate_up_count = 0
    down_count = 0
    num_experts = 0

    for k in keys:
        if not k.startswith(layer_prefix):
            continue

        if "mlp.experts.gate_up_proj" in k:
            fused_weight = weights.pop(k)  # [num_experts, 2*inter_size, hidden_size]
            num_experts = fused_weight.shape[0]

            prefix = k.rsplit(".gate_up_proj", 1)[0]

            gate_weight = fused_weight[:, :moe_intermediate_size, :]
            up_weight = fused_weight[:, moe_intermediate_size:, :]

            for expert_idx in range(num_experts):
                weights[f"{prefix}.{expert_idx}.gate_proj.weight"] = gate_weight[expert_idx]
                weights[f"{prefix}.{expert_idx}.up_proj.weight"] = up_weight[expert_idx]

            gate_up_count += 1

        elif "mlp.experts.down_proj" in k:
            down_weight = weights.pop(k)  # [num_experts, hidden_size, inter_size]
            num_experts = down_weight.shape[0]

            prefix = k.rsplit(".down_proj", 1)[0]

            for expert_idx in range(num_experts):
                weights[f"{prefix}.{expert_idx}.down_proj.weight"] = down_weight[expert_idx]

            down_count += 1

    # if gate_up_count > 0 or down_count > 0:
    #     logger.debug(
    #         f"Layer {layer_num}: Split {gate_up_count} fused gate_up_proj weights, "
    #         f"{down_count} down_proj weights into {num_experts} experts each"
    #     )


class Qwen35NextFullAttentionTransformerLayerWeight(Qwen3NextFullAttentionTransformerLayerWeight):
    def load_hf_weights(self, weights):
        self._split_fused_expert_weights(weights)
        super().load_hf_weights(weights)

    def _split_fused_expert_weights(self, weights):
        moe_intermediate_size = self.network_config_.get("moe_intermediate_size")
        if moe_intermediate_size is None:
            moe_intermediate_size = self.network_config_.get("intermediate_size")

        if moe_intermediate_size is None:
            logger.warning(
                f"Layer {self.layer_num_}: Cannot find moe_intermediate_size in config, "
                "skipping fused expert weight splitting"
            )
            return

        layer_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        has_fused_weights = any(layer_prefix in k and ("gate_up_proj" in k or "down_proj" in k) for k in weights.keys())

        if has_fused_weights:
            split_fused_expert_weights(weights, self.layer_num_, moe_intermediate_size)


class Qwen35NextGatedDeltaNetTransformerLayerWeight(Qwen3NextGatedDeltaNetTransformerLayerWeight):
    """Qwen3.5 GDN weights.

    vLLM loads Qwen3.5 linear-attention projections directly from grouped checkpoints:
    - in_proj_qkv.weight ([all_q, all_k, all_v])
    - in_proj_z.weight
    - in_proj_b.weight
    - in_proj_a.weight

    To preserve correct TP row slicing in LightLLM, we split in_proj_qkv into three
    virtual weights (in_proj_q / in_proj_k / in_proj_v) during preprocessing and load
    them as separate ROWMM shards.
    """

    def _init_gdn_weight(self):
        # Initialize everything from parent first, then override only linear_in_proj.
        super()._init_gdn_weight()

        prefix = f"model.layers.{self.layer_num_}.linear_attn"
        hidden_size = self.network_config_["hidden_size"]
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim

        # NOTE: keep grouped layout directly (q, k, v, z, b, a).
        self.linear_in_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[
                qk_dim,
                qk_dim,
                v_dim,
                v_dim,
                self.linear_num_v_heads,
                self.linear_num_v_heads,
            ],
            weight_names=[
                f"{prefix}.in_proj_q.weight",
                f"{prefix}.in_proj_k.weight",
                f"{prefix}.in_proj_v.weight",
                f"{prefix}.in_proj_z.weight",
                f"{prefix}.in_proj_b.weight",
                f"{prefix}.in_proj_a.weight",
            ],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("in_proj_weight"),
        )

    def load_hf_weights(self, weights):
        self._split_fused_expert_weights(weights)
        super().load_hf_weights(weights)

    def _preprocess_weight(self, weights):
        # Keep parent conv1d preprocessing path.
        linear_conv1d_weight_name = f"model.layers.{self.layer_num_}.linear_attn.conv1d.weight"
        linear_conv1d_bias_name = f"model.layers.{self.layer_num_}.linear_attn.conv1d.bias"

        if linear_conv1d_weight_name in weights:
            weights[linear_conv1d_weight_name] = self._parse_linear_conv1d(
                weights[linear_conv1d_weight_name].squeeze(1)
            )
        if linear_conv1d_bias_name in weights:
            weights[linear_conv1d_bias_name] = self._parse_linear_conv1d(weights[linear_conv1d_bias_name])

        self._split_linear_in_proj_qkv(weights)

    def _split_linear_in_proj_qkv(self, weights):
        prefix = f"model.layers.{self.layer_num_}.linear_attn"
        qkv_name = f"{prefix}.in_proj_qkv.weight"
        if qkv_name not in weights:
            return

        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim
        expected_rows = 2 * qk_dim + v_dim

        qkv = weights[qkv_name]
        if qkv.shape[0] != expected_rows:
            logger.warning(
                f"Layer {self.layer_num_}: unexpected in_proj_qkv shape "
                f"{tuple(qkv.shape)}, expected first dim {expected_rows}; skip split"
            )
            return

        q, k, v = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)
        weights[f"{prefix}.in_proj_q.weight"] = q
        weights[f"{prefix}.in_proj_k.weight"] = k
        weights[f"{prefix}.in_proj_v.weight"] = v
        del weights[qkv_name]

    def _split_fused_expert_weights(self, weights):
        moe_intermediate_size = self.network_config_.get("moe_intermediate_size")
        if moe_intermediate_size is None:
            moe_intermediate_size = self.network_config_.get("intermediate_size")

        if moe_intermediate_size is None:
            logger.warning(
                f"Layer {self.layer_num_}: Cannot find moe_intermediate_size in config, "
                "skipping fused expert weight splitting"
            )
            return

        layer_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        has_fused_weights = any(layer_prefix in k and ("gate_up_proj" in k or "down_proj" in k) for k in weights.keys())

        if has_fused_weights:
            split_fused_expert_weights(weights, self.layer_num_, moe_intermediate_size)
