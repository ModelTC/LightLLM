import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight


class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.is_moe = (
            self.network_config_["n_routed_experts"] is not None
            and self.layer_num_ >= self.network_config_["first_k_dense_replace"]
            and self.layer_num_ % self.network_config_["moe_layer_freq"] == 0
        )
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.n_routed_experts = self.network_config_["n_routed_experts"]
        self.q_lora_rank = self.network_config_["q_lora_rank"]
        self.qk_nope_head_dim = self.network_config_["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.network_config_["qk_rope_head_dim"]
        self.num_attention_heads = self.network_config_["num_attention_heads"]
        self.kv_lora_rank = self.network_config_["kv_lora_rank"]
        self.experts_up_proj = [None] * self.n_routed_experts
        self.experts_gate_proj = [None] * self.n_routed_experts
        self.w2_list = [None] * self.n_routed_experts
        return

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.kv_a_proj_with_mqa_,
            self.kv_a_layernorm_,
            self.fuse_vo_weight_,
            self.ffn_norm_weight_,
        ]
        if self.q_lora_rank is not None:
            weights += [self.q_a_proj_, self.q_a_layernorm_, self.fuse_qk_weight_, self.q_rope_proj_]
        else:
            weights += [self.fuse_qk_weight_, self.q_rope_proj_]
        if self.is_moe:
            weights += [self.moe_gate, self.w1, self.w2]
            if self.network_config_["n_shared_experts"] is not None:
                weights += [self.gate_up_proj, self.down_proj]
        else:
            weights += [
                self.gate_up_proj,
                self.down_proj,
            ]

        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        q_split_n_embed_with_rope = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads // self.world_size_
        )

        # q k v weights for llama
        if self.q_lora_rank is None:
            if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
                self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"]
                self.q_weight_ = self.q_weight_[
                    q_split_n_embed_with_rope * self.tp_rank_ : q_split_n_embed_with_rope * (self.tp_rank_ + 1), :
                ]
                self.q_weight_ = self.q_weight_.transpose(0, 1).contiguous().to(self.data_type_).cpu()
        else:
            if f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight" in weights:
                q_a_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight"]
                self.q_a_proj_ = self._cuda(q_a_proj_.transpose(0, 1))

            if f"model.layers.{self.layer_num_}.self_attn.q_a_layernorm.weight" in weights:
                q_a_layernorm_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_a_layernorm.weight"]
                self.q_a_layernorm_ = self._cuda(q_a_layernorm_)

            if f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight" in weights:
                q_b_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight"]
                q_b_proj_ = q_b_proj_[
                    q_split_n_embed_with_rope * self.tp_rank_ : q_split_n_embed_with_rope * (self.tp_rank_ + 1), :
                ]
                self.q_b_proj_ = q_b_proj_.transpose(0, 1).contiguous().to(self.data_type_).cpu()

        if f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight" in weights:
            kv_a_proj_with_mqa_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight"]
            self.kv_a_proj_with_mqa_ = self._cuda(kv_a_proj_with_mqa_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.kv_a_layernorm.weight" in weights:
            kv_a_layernorm_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_a_layernorm.weight"]
            self.kv_a_layernorm_ = self._cuda(kv_a_layernorm_)

        if f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight" in weights:
            kv_b_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight"]
            k_b_proj_ = kv_b_proj_.view(self.num_attention_heads, self.qk_nope_head_dim * 2, self.kv_lora_rank)[
                :, : self.qk_nope_head_dim, :
            ]
            v_b_proj_ = kv_b_proj_.T.view(
                self.kv_lora_rank,
                self.num_attention_heads,
                self.qk_nope_head_dim * 2,
            )[:, :, self.qk_nope_head_dim :]
            self.k_b_proj_ = (
                k_b_proj_[self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :]
                .contiguous()
                .to(self.data_type_)
                .cpu()
            )
            self.v_b_proj_ = (
                v_b_proj_.transpose(0, 1)[
                    self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :
                ]
                .contiguous()
                .to(self.data_type_)
                .cpu()
            )

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            o_weight_ = o_weight_.T.view(self.num_attention_heads, self.qk_nope_head_dim, -1)
            print("Here", self.tp_q_head_num_, self.world_size_, self.network_config_["num_attention_heads"])
            self.o_weight_ = (
                o_weight_[self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :]
                .contiguous()
                .to(self.data_type_)
                .cpu()
            )

        with self.lock:
            if hasattr(self, "q_weight_") and hasattr(self, "k_b_proj_"):
                hidden_size = self.network_config_["hidden_size"]
                q_nope_proj_, q_rope_proj_ = torch.split(
                    self.q_weight_.view(hidden_size, -1, self.qk_nope_head_dim + self.qk_rope_head_dim),
                    [self.qk_nope_head_dim, self.qk_rope_head_dim],
                    dim=-1,
                )
                self.q_rope_proj_ = self._cuda(q_rope_proj_.reshape(hidden_size, -1))
                q_nope_proj_ = q_nope_proj_.unsqueeze(2).to(torch.float64)

                k_nope_proj_ = self.k_b_proj_.unsqueeze(0)
                k_nope_proj_ = k_nope_proj_.to(torch.float64)

                self.fuse_qk_weight_ = self._cuda(
                    torch.matmul(q_nope_proj_, k_nope_proj_).view(hidden_size, self.tp_q_head_num_ * self.kv_lora_rank)
                )

                delattr(self, "q_weight_")
                delattr(self, "k_b_proj_")

        with self.lock:
            if hasattr(self, "q_b_proj_") and hasattr(self, "k_b_proj_"):
                q_nope_proj_, q_rope_proj_ = torch.split(
                    self.q_b_proj_.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
                    [self.qk_nope_head_dim, self.qk_rope_head_dim],
                    dim=-1,
                )
                self.q_rope_proj_ = self._cuda(q_rope_proj_.reshape(-1, self.qk_rope_head_dim * self.tp_q_head_num_))
                q_nope_proj_ = q_nope_proj_.unsqueeze(2).to(torch.float64)

                k_nope_proj_ = self.k_b_proj_.unsqueeze(0)
                k_nope_proj_ = k_nope_proj_.to(torch.float64)

                self.fuse_qk_weight_ = self._cuda(
                    torch.matmul(q_nope_proj_, k_nope_proj_).view(-1, self.tp_q_head_num_ * self.kv_lora_rank)
                )

                delattr(self, "q_b_proj_")
                delattr(self, "k_b_proj_")

        with self.lock:
            if hasattr(self, "v_b_proj_") and hasattr(self, "o_weight_"):
                self.fuse_vo_weight_ = self._cuda(
                    torch.matmul(self.v_b_proj_.to(torch.float64), self.o_weight_.to(torch.float64)).view(
                        -1, self.network_config_["hidden_size"]
                    )
                )

                delattr(self, "v_b_proj_")
                delattr(self, "o_weight_")

        return

    def _load_mlp(self, split_inter_size, weights, mlp_prefix):
        if f"{mlp_prefix}.up_proj.weight" in weights:
            up_proj = weights[f"{mlp_prefix}.up_proj.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.up_proj = up_proj.transpose(0, 1)

        if f"{mlp_prefix}.gate_proj.weight" in weights:
            gate_proj = weights[f"{mlp_prefix}.gate_proj.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.gate_proj = gate_proj.transpose(0, 1)

        self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=1)

        if f"{mlp_prefix}.down_proj.weight" in weights:
            self.down_proj = weights[f"{mlp_prefix}.down_proj.weight"][
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(
                weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"]
            )

        if self.is_moe:
            if f"model.layers.{self.layer_num_}.mlp.gate.weight" in weights:
                moe_gate = weights[f"model.layers.{self.layer_num_}.mlp.gate.weight"]
                self.moe_gate = self._cuda(moe_gate.transpose(0, 1))

            shared_intermediate_size = (
                self.network_config_["moe_intermediate_size"] * self.network_config_["n_shared_experts"]
            )
            shared_split_inter_size = shared_intermediate_size // self.world_size_
            self._load_mlp(shared_split_inter_size, weights, f"model.layers.{self.layer_num_}.mlp.shared_experts")

            split_inter_size = self.network_config_["moe_intermediate_size"] // self.world_size_
            for i_experts in range(self.n_routed_experts):
                expert_up_proj = None
                expert_gate_proj = None
                expert_gate_up_proj = None
                expert_down_proj = None

                if f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.up_proj.weight" in weights:
                    expert_up_proj = weights[f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.up_proj.weight"][
                        split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
                    ]
                    self.experts_up_proj[i_experts] = expert_up_proj

                if f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.gate_proj.weight" in weights:
                    expert_gate_proj = weights[
                        f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.gate_proj.weight"
                    ][split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
                    self.experts_gate_proj[i_experts] = expert_gate_proj

                # if expert_gate_proj is not None and expert_up_proj is not None:
                #     expert_gate_up_proj = torch.cat([expert_gate_proj, expert_up_proj], dim=0)
                #     expert_gate_up_proj = self._cuda(expert_gate_up_proj)

                if f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.down_proj.weight" in weights:
                    expert_down_proj = weights[
                        f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.down_proj.weight"
                    ][:, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)]
                    expert_down_proj = self._cuda(expert_down_proj)
                    self.w2_list[i_experts] = expert_down_proj
            with self.lock:
                if (
                    hasattr(self, "experts_up_proj")
                    and None not in self.experts_up_proj
                    and None not in self.experts_gate_proj
                    and None not in self.w2_list
                ):

                    w1_list = []
                    for i_experts in range(self.n_routed_experts):
                        expert_gate_up_proj = torch.cat(
                            [self.experts_gate_proj[i_experts], self.experts_up_proj[i_experts]], dim=0
                        )
                        expert_gate_up_proj = self._cuda(expert_gate_up_proj)
                        w1_list.append(expert_gate_up_proj)

                    inter_shape, hidden_size = w1_list[0].shape[0], w1_list[0].shape[1]
                    self.w1 = torch._utils._flatten_dense_tensors(w1_list).view(len(w1_list), inter_shape, hidden_size)
                    inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                    self.w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(
                        len(self.w2_list), inter_shape, hidden_size
                    )

                    delattr(self, "w2_list")
                    delattr(self, "experts_up_proj")
                    delattr(self, "experts_gate_proj")
        else:
            inter_size = self.network_config_["intermediate_size"]
            split_inter_size = inter_size // self.world_size_

            self._load_mlp(split_inter_size, weights, f"model.layers.{self.layer_num_}.mlp")
        return
