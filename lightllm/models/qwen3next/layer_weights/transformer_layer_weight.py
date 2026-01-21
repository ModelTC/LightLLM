import torch
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    NormWeight,
    TpParameterWeight,
)
from typing_extensions import override


class Qwen3NextFullAttentionTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    @override
    def _init_weight(self):
        super()._init_weight()
        # Additional architecture
        self._init_o_gate_proj_weight()
        self._init_gate_shared_expert_weight()
        return

    @override
    def load_hf_weights(self, weights):
        self._split_q_with_gate(weights)
        super().load_hf_weights(weights)

    def _init_o_gate_proj_weight(self):
        self._o_gate_weight_name = f"model.layers.{self.layer_num_}.self_attn.o_gate_proj.weight"
        self.o_gate_proj = ROWMMWeight(
            weight_names=self._o_gate_weight_name,
            data_type=self.data_type_,
            bias_names=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="o_gate_proj",
        )

    def _init_gate_shared_expert_weight(self):
        prefix = f"model.layers.{self.layer_num_}.mlp.shared_expert"
        self.shared_expert_gate_up_proj = ROWMMWeight(
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate_up_proj",
        )
        self.shared_expert_down_proj = COLMMWeight(
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_down_proj",
        )
        self.shared_expert_gate = ROWMMWeight(
            weight_names=f"model.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate",
            tp_rank=0,
            tp_world_size=1,
        )

    def _split_q_with_gate(self, weights):
        if self._q_weight_name in weights:
            weight = weights[self._q_weight_name]
            num_heads = self.tp_q_head_num_ * self.tp_world_size_
            weight = weight.view(num_heads * 2, self.head_dim, -1)
            _q_proj = weight[0::2].reshape(-1, weight.shape[-1])
            _gate_proj = weight[1::2].reshape(-1, weight.shape[-1])
            weights[self._q_weight_name] = _q_proj
            weights[self._o_gate_weight_name] = _gate_proj


class Qwen3NextGatedDeltaNetTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg):
        self.is_moe = (
            network_config["num_experts"] > 0
            and layer_num not in network_config["mlp_only_layers"]
            and (layer_num + 1) % network_config["decoder_sparse_step"] == 0
        )
        super().__init__(layer_num, data_type, network_config, quant_cfg)

    @override
    def _parse_config(self):
        self.linear_num_v_heads = self.network_config_["linear_num_value_heads"]
        self.linear_num_k_heads = self.network_config_["linear_num_key_heads"]
        self.linear_k_head_dim = self.network_config_["linear_key_head_dim"]
        self.linear_v_head_dim = self.network_config_["linear_value_head_dim"]

    @override
    def _init_weight(self):
        self.att_norm_weight_ = NormWeight(
            self._att_norm_weight_name, self.data_type_, bias_name=self._att_norm_bias_name
        )
        self._init_gdn_weight()
        self.ffn_norm_weight_ = NormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_gate_shared_expert_weight()

    def _init_gdn_weight(self):
        prefix = f"model.layers.{self.layer_num_}.linear_attn"
        self.linear_conv1d = COLMMWeight(
            weight_names=f"{prefix}.conv1d.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="conv1d_weight",
        )

        self.linear_in_proj = ROWMMWeight(
            weight_names=[f"{prefix}.in_proj_qkvz.weight", f"{prefix}.in_proj_ba.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="in_proj_weight",
        )

        self.linear_out_proj = COLMMWeight(
            weight_names=f"{prefix}.out_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="out_proj_weight",
        )

        split_n_embed = self.linear_num_v_heads // self.tp_world_size_
        self.linear_dt_bias = TpParameterWeight(
            weight_name=f"{prefix}.dt_bias",
            data_type=torch.float32,
            split_n_embed=split_n_embed,
            bias_name=None,
            weight_shape=(self.linear_num_v_heads,),  # Full shape before TP split
            bias_shape=None,
        )

        self.linear_A_log = TpParameterWeight(
            weight_name=f"{prefix}.A_log",
            data_type=torch.float32,
            split_n_embed=split_n_embed,
            bias_name=None,
            weight_shape=(self.linear_num_v_heads,),  # Full shape before TP split
            bias_shape=None,
        )

        self.linear_norm = NormWeight(
            weight_name=f"{prefix}.norm.weight",
            data_type=self.data_type_,
        )

    @override
    def load_hf_weights(self, weights):
        self._preprocess_weight(weights)
        return super().load_hf_weights(weights)

    def _preprocess_weight(self, weights):
        linear_conv1d_weight_name = f"model.layers.{self.layer_num_}.linear_attn.conv1d.weight"
        linear_conv1d_bias_name = f"model.layers.{self.layer_num_}.linear_attn.conv1d.bias"
        if linear_conv1d_weight_name in weights:
            weights[linear_conv1d_weight_name] = self._parse_linear_conv1d(
                weights[linear_conv1d_weight_name].squeeze(1)
            ).transpose(0, 1)
        if linear_conv1d_bias_name in weights:
            weights[linear_conv1d_bias_name] = self._parse_linear_conv1d(weights[linear_conv1d_bias_name])

    def _parse_linear_conv1d(self, weight):
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim
        q_part, k_part, v_part = torch.split(weight, [qk_dim, qk_dim, v_dim], dim=0)
        q_splits = q_part.chunk(self.tp_world_size_, dim=0)
        k_splits = k_part.chunk(self.tp_world_size_, dim=0)
        v_splits = v_part.chunk(self.tp_world_size_, dim=0)
        new_weight = torch.cat(
            [torch.cat([q_splits[i], k_splits[i], v_splits[i]], dim=0) for i in range(self.tp_world_size_)], dim=0
        )
        return new_weight

    def _init_gate_shared_expert_weight(self):
        prefix = f"model.layers.{self.layer_num_}.mlp.shared_expert"
        self.shared_expert_gate_up_proj = ROWMMWeight(
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate_up_proj",
        )
        self.shared_expert_down_proj = COLMMWeight(
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_down_proj",
        )
        self.shared_expert_gate = ROWMMWeight(
            weight_names=f"model.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate",
            tp_rank=0,
            tp_world_size=1,
        )
