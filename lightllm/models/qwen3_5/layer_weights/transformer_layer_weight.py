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
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim

        # The attention kernels consume [q, k, v, z, b, a].
        self._init_gdn_in_proj(
            out_dims=[qk_dim, qk_dim, v_dim, v_dim, self.linear_num_v_heads, self.linear_num_v_heads],
            weight_names=[f"{prefix}.in_proj_{part}.weight" for part in ("q", "k", "v", "z", "b", "a")],
            quant_method=self.get_quant_method("in_proj_weight"),
        )

    def _init_gdn_in_proj(self, out_dims, weight_names, quant_method):
        self.linear_in_proj = ROWMMWeight(
            in_dim=self.network_config_["hidden_size"],
            out_dims=out_dims,
            weight_names=weight_names,
            data_type=self.data_type_,
            quant_method=quant_method,
        )

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
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim
        quant_method = self.linear_in_proj.quant_method
        layouts = [("weight", 0, 1)]
        if quant_method.method_name in ("awq", "awq_marlin"):
            pack_factor = quant_method.pack_factor
            layouts.extend([("qweight", -1, pack_factor), ("qzeros", -1, pack_factor), ("scales", -1, 1)])

        for suffix, axis, factor in layouts:
            qkv_name = f"{prefix}.in_proj_qkv.{suffix}"
            qkv = weights.get(qkv_name)
            if qkv is None:
                continue
            sizes = [qk_dim // factor, qk_dim // factor, v_dim // factor]
            if qkv.shape[axis] != sum(sizes):
                logger.warning(
                    f"Layer {self.layer_num_}: unexpected {qkv_name} shape "
                    f"{tuple(qkv.shape)}, expected dimension {axis} to be {sum(sizes)}; skip split"
                )
                continue
            for part, tensor in zip(("q", "k", "v"), torch.split(qkv, sizes, dim=axis)):
                weights[f"{prefix}.in_proj_{part}.{suffix}"] = tensor
            del weights[qkv_name]
