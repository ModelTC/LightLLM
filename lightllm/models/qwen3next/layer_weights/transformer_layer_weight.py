import torch
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    RMSNormWeight,
    TpParameterWeight,
    KVROWNMMWeight,
)
from typing_extensions import override


class Qwen3NextFullAttentionTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    @override
    def _init_qkv(self):
        # Override parent's QKVROWNMMWeight which requires kv_head_num % tp == 0.
        # Qwen3-Next has very few KV heads (e.g., 2) so we use separate q + kv weights.
        # KVROWNMMWeight handles the kv_head_num < tp_world_size case via repeating.
        in_dim = self.n_embed
        q_out_dim = self.q_head_num_ * self.head_dim
        # Define o_gate weight name here (used by _split_q_with_gate during load)
        self._o_gate_weight_name = f"model.layers.{self.layer_num_}.self_attn.o_gate_proj.weight"
        # Fused Q + gate projection: single GEMM outputs [q, gate] concatenated
        self.q_gate_proj = ROWMMWeight(
            in_dim=in_dim,
            out_dims=[q_out_dim, q_out_dim],
            weight_names=[self._q_weight_name, self._o_gate_weight_name],
            data_type=self.data_type_,
            bias_names=None,
            quant_method=self.get_quant_method("q_proj"),
        )
        self.kv_proj = KVROWNMMWeight(
            in_dim=in_dim,
            kv_head_num=self.k_head_num_,
            head_dim=self.head_dim,
            weight_names=[self._k_weight_name, self._v_weight_name],
            data_type=self.data_type_,
            bias_names=[self._k_bias_name, self._v_bias_name],
            quant_method=self.get_quant_method("kv_proj"),
        )

    @override
    def _init_weight(self):
        super()._init_weight()
        # Additional architecture (o_gate is now fused into q_gate_proj in _init_qkv)
        self._init_gate_shared_expert_weight()
        return

    @override
    def load_hf_weights(self, weights):
        self._split_q_with_gate(weights)
        super().load_hf_weights(weights)

    def _split_q_with_gate(self, weights):
        if self._q_weight_name in weights:
            weight = weights[self._q_weight_name]
            num_heads = self.q_head_num_
            weight = weight.view(num_heads * 2, self.head_dim, -1)
            _q_proj = weight[0::2].reshape(-1, weight.shape[-1])
            _gate_proj = weight[1::2].reshape(-1, weight.shape[-1])
            weights[self._q_weight_name] = _q_proj
            weights[self._o_gate_weight_name] = _gate_proj

    def _init_gate_shared_expert_weight(self):
        prefix = f"model.layers.{self.layer_num_}.mlp.shared_expert"
        hidden_size = self.network_config_["hidden_size"]
        shared_inter = self.network_config_["shared_expert_intermediate_size"]
        self.shared_expert_gate_up_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[shared_inter, shared_inter],
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_expert_gate_up_proj"),
        )
        self.shared_expert_down_proj = COLMMWeight(
            in_dim=shared_inter,
            out_dims=[hidden_size],
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_expert_down_proj"),
        )
        self.shared_expert_gate = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[1],
            weight_names=f"model.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )


class Qwen3NextGatedDeltaNetTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        self.is_moe = (
            network_config["num_experts"] > 0
            and layer_num not in network_config["mlp_only_layers"]
            and (layer_num + 1) % network_config["decoder_sparse_step"] == 0
        )
        super().__init__(layer_num, data_type, network_config, quant_cfg)

    @override
    def _parse_config(self):
        super()._parse_config()
        self.linear_num_v_heads = self.network_config_["linear_num_value_heads"]
        self.linear_num_k_heads = self.network_config_["linear_num_key_heads"]
        self.linear_k_head_dim = self.network_config_["linear_key_head_dim"]
        self.linear_v_head_dim = self.network_config_["linear_value_head_dim"]

    @override
    def _init_weight(self):
        hidden_size = self.network_config_["hidden_size"]
        self.att_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._att_norm_weight_name,
            data_type=self.data_type_,
        )
        self._init_gdn_weight()
        self.ffn_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=self._ffn_norm_weight_name,
            data_type=self.data_type_,
        )
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_gate_shared_expert_weight()

    def _init_gdn_weight(self):
        prefix = f"model.layers.{self.layer_num_}.linear_attn"
        hidden_size = self.network_config_["hidden_size"]
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim
        conv1d_channels = qk_dim + qk_dim + v_dim  # q + k + v concatenated
        kernel_size = self.network_config_.get("linear_conv_kernel_dim", 4)

        # Conv1d weight: after _preprocess_weight, shape is [channels, kernel_size].
        # ROWMMWeight row-slices out_dims (rows), matching TP split of channels dim.
        # causal_conv1d_fn expects weight shape (dim, width) = (channels_per_tp, kernel_size).
        self.linear_conv1d = ROWMMWeight(
            in_dim=kernel_size,
            out_dims=[conv1d_channels],
            weight_names=f"{prefix}.conv1d.weight",
            data_type=self.data_type_,
            quant_method=None,
        )

        # in_proj_qkvz: q(qk_dim) + k(qk_dim) + v(v_dim) + z(v_dim)
        # in_proj_ba: beta(num_v_heads) + alpha(num_v_heads) — per-head scalars
        qkvz_dim = qk_dim + qk_dim + v_dim + v_dim
        ba_dim = self.linear_num_v_heads + self.linear_num_v_heads
        self.linear_in_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[qkvz_dim, ba_dim],
            weight_names=[f"{prefix}.in_proj_qkvz.weight", f"{prefix}.in_proj_ba.weight"],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("in_proj_weight"),
        )

        self.linear_out_proj = COLMMWeight(
            in_dim=v_dim,
            out_dims=[hidden_size],
            weight_names=f"{prefix}.out_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("out_proj_weight"),
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

        # Norm is applied per-head across head_dim, not across all heads
        linear_norm_dim = self.linear_v_head_dim
        self.linear_norm = RMSNormWeight(
            dim=linear_norm_dim,
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
            # squeeze [channels, 1, kernel] -> [channels, kernel], then rearrange for TP
            # Result shape: [channels, kernel_size] — matches causal_conv1d_fn's (dim, width)
            weights[linear_conv1d_weight_name] = self._parse_linear_conv1d(
                weights[linear_conv1d_weight_name].squeeze(1)
            )
        if linear_conv1d_bias_name in weights:
            weights[linear_conv1d_bias_name] = self._parse_linear_conv1d(weights[linear_conv1d_bias_name])
        self._rearrange_gdn_in_proj_weights(weights)

    def _rearrange_gdn_in_proj_weights(self, weights):
        """Rearrange in_proj_qkvz and in_proj_ba weight rows from interleaved per-k-head layout
        to TP-aware grouped layout so that after ROWMMWeight's row-slicing, each rank's
        MM output is already [q_chunk, k_chunk, v_chunk, z_chunk, b_chunk, a_chunk].

        This eliminates the expensive split+reshape+cat in _fix_query_key_value_ba_ordering
        at inference time, replacing it with simple slicing.

        The key challenge is that ROWMMWeight slices each weight as a contiguous row chunk
        (rows [start:end]). So we arrange the rows such that each TP chunk contains
        the grouped layout for that rank:
        1. Deinterleave from per-k-head groups into per-component tensors
        2. Chunk each component by TP
        3. Reassemble as [q_tp0, k_tp0, v_tp0, z_tp0, q_tp1, k_tp1, ...] so row-slicing
           gives each rank [q_chunk, k_chunk, v_chunk, z_chunk].
        Same pattern as _parse_linear_conv1d uses for conv1d weights.
        """
        num_k = self.linear_num_k_heads
        k_dim = self.linear_k_head_dim
        v_dim = self.linear_v_head_dim
        num_v_per_k = self.linear_num_v_heads // num_k
        tp = self.tp_world_size_

        # Rearrange in_proj_qkvz
        qkvz_name = f"model.layers.{self.layer_num_}.linear_attn.in_proj_qkvz.weight"
        if qkvz_name in weights:
            w = weights[qkvz_name]
            hidden = w.shape[-1]
            # Each k-head group: q(k_dim) + k(k_dim) + v(num_v_per_k * v_dim) + z(num_v_per_k * v_dim) rows
            group_size = k_dim + k_dim + num_v_per_k * v_dim + num_v_per_k * v_dim
            w = w.view(num_k, group_size, hidden)
            v_block = num_v_per_k * v_dim
            all_q = w[:, :k_dim, :].reshape(-1, hidden)  # [total_q_dim, H]
            all_k = w[:, k_dim : 2 * k_dim, :].reshape(-1, hidden)  # [total_k_dim, H]
            all_v = w[:, 2 * k_dim : 2 * k_dim + v_block, :].reshape(-1, hidden)  # [total_v_dim, H]
            all_z = w[:, 2 * k_dim + v_block :, :].reshape(-1, hidden)  # [total_v_dim, H]
            # Chunk each component by TP, interleave so row-slicing gives grouped layout per rank
            q_chunks = all_q.chunk(tp, dim=0)
            k_chunks = all_k.chunk(tp, dim=0)
            v_chunks = all_v.chunk(tp, dim=0)
            z_chunks = all_z.chunk(tp, dim=0)
            weights[qkvz_name] = torch.cat(
                [torch.cat([q_chunks[i], k_chunks[i], v_chunks[i], z_chunks[i]], dim=0) for i in range(tp)],
                dim=0,
            )

        # Rearrange in_proj_ba
        ba_name = f"model.layers.{self.layer_num_}.linear_attn.in_proj_ba.weight"
        if ba_name in weights:
            w = weights[ba_name]
            hidden = w.shape[-1]
            group_size = 2 * num_v_per_k
            w = w.view(num_k, group_size, hidden)
            all_b = w[:, :num_v_per_k, :].reshape(-1, hidden)  # [total_num_v, H]
            all_a = w[:, num_v_per_k:, :].reshape(-1, hidden)  # [total_num_v, H]
            b_chunks = all_b.chunk(tp, dim=0)
            a_chunks = all_a.chunk(tp, dim=0)
            weights[ba_name] = torch.cat(
                [torch.cat([b_chunks[i], a_chunks[i]], dim=0) for i in range(tp)],
                dim=0,
            )

    def _parse_linear_conv1d(self, weight):
        qk_dim = self.linear_num_k_heads * self.linear_k_head_dim
        v_dim = self.linear_num_v_heads * self.linear_v_head_dim
        q_bias, k_bias, v_bias = torch.split(weight, [qk_dim, qk_dim, v_dim], dim=0)
        q_splits = q_bias.chunk(self.tp_world_size_, dim=0)
        k_splits = k_bias.chunk(self.tp_world_size_, dim=0)
        v_splits = v_bias.chunk(self.tp_world_size_, dim=0)
        new_weight = torch.cat(
            [torch.cat([q_splits[i], k_splits[i], v_splits[i]], dim=0) for i in range(self.tp_world_size_)], dim=0
        )
        return new_weight

    def _init_gate_shared_expert_weight(self):
        prefix = f"model.layers.{self.layer_num_}.mlp.shared_expert"
        hidden_size = self.network_config_["hidden_size"]
        shared_inter = self.network_config_["shared_expert_intermediate_size"]
        self.shared_expert_gate_up_proj = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[shared_inter, shared_inter],
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_expert_gate_up_proj"),
        )
        self.shared_expert_down_proj = COLMMWeight(
            in_dim=shared_inter,
            out_dims=[hidden_size],
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("shared_expert_down_proj"),
        )
        self.shared_expert_gate = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[1],
            weight_names=f"model.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
