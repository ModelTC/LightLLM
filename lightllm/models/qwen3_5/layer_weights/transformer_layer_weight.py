import torch

from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextTransformerLayerWeight,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen35TransformerLayerWeight(Qwen3NextTransformerLayerWeight):
    def _init_weight_names(self):
        super()._init_weight_names()
        self._gate_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"model.layers.{self.layer_num_}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._gate_up_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"
        self._gate_up_bias_name = None
        self._down_weight_name = f"model.layers.{self.layer_num_}.mlp.down_proj.weight"
        self._down_bias_name = None

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

    def _split_q_with_gate(self, weights):
        # The official Qwen3.5 base checkpoint packs o_gate_proj into q_proj with an
        # interleaved per-head layout, so the parent de-interleaves it. Fine-tunes
        # re-saved through standard HF transformers store o_gate_proj as a separate
        # tensor; in that case the q_proj weight is already plain Q and de-interleaving
        # would scramble its rows. Detect the separate layout and skip the split.
        if self._o_gate_weight_name in weights:
            return
        if self._q_weight_name not in weights:
            return
        q_weight = weights[self._q_weight_name]
        expected_packed_rows = self.q_head_num_ * self.head_dim * 2
        if q_weight.shape[0] != expected_packed_rows:
            # Already plain Q (rows == q_head_num * head_dim) — nothing to split.
            return
        super()._split_q_with_gate(weights)

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
