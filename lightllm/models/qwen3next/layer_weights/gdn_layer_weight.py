from typing_extensions import override
import torch

from lightllm.common.basemodel.layer_weights.transformer_layer_weight import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    MultiROWMMWeight,
    COLMMWeight,
    NormWeight,
    ROWBMMWeight,
    TpParameterWeight,
)


class Qwen3NextGatedDeltaNetWeight(TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode, quant_cfg):
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def load_hf_weights(self, weights):
        if self.conv1d.weight_name in weights:
            weights[self.conv1d.weight_name] = self._parse_conv1d(weights[self.conv1d.weight_name].squeeze(1))
        if self.conv1d.bias_name in weights:
            weights[self.conv1d.bias_name] = self._parse_conv1d(weights[self.conv1d.bias_name])
        super().load_hf_weights(weights)

    @override
    def _parse_config(self):
        self.num_v_heads = self.network_config_["linear_num_value_heads"]
        self.num_k_heads = self.network_config_["linear_num_key_heads"]
        self.k_head_dim = self.network_config_["linear_key_head_dim"]
        self.v_head_dim = self.network_config_["linear_value_head_dim"]

    @override
    def _init_weight(self):
        prefix = f"model.layers.{self.layer_num_}.linear_attn"
        self.conv1d = ROWMMWeight(
            weight_name=f"{prefix}.conv1d.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="conv1d_weight",
        )

        self.in_proj = MultiROWMMWeight(
            weight_names=[f"{prefix}.in_proj_qkvz.weight", f"{prefix}.in_proj_ba.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="in_proj_weight",
        )

        self.out_proj = COLMMWeight(
            weight_name=f"{prefix}.out_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="out_proj_weight",
        )

        self.dt_bias = TpParameterWeight(
            weight_name=f"{prefix}.dt_bias",
            data_type=torch.float32,
            split_n_embed=self.num_v_heads // self.tp_world_size_,
        )

        self.A_log = TpParameterWeight(
            weight_name=f"{prefix}.A_log",
            data_type=torch.float32,
            split_n_embed=self.num_v_heads // self.tp_world_size_,
        )

        self.norm = NormWeight(
            weight_name=f"{prefix}.norm.weight",
            data_type=self.data_type_,
        )

    def _parse_conv1d(self, weight):
        qk_dim = self.num_k_heads * self.k_head_dim
        v_dim = self.num_v_heads * self.v_head_dim

        q_bias, k_bias, v_bias = torch.split(weight, [qk_dim, qk_dim, v_dim], dim=0)
        q_splits = q_bias.chunk(self.tp_world_size_, dim=0)
        k_splits = k_bias.chunk(self.tp_world_size_, dim=0)
        v_splits = v_bias.chunk(self.tp_world_size_, dim=0)

        new_weight = torch.cat(
            [torch.cat([q_splits[i], k_splits[i], v_splits[i]], dim=0) for i in range(self.tp_world_size_)], dim=0
        )

        return new_weight
