import os
import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    ROWMMWeightNoTP,
    MultiROWMMWeight,
    MultiROWMMWeightNoTP,
    COLMMWeight,
    COLMMWeightNoTp,
    MultiCOLMMWeight,
    MultiCOLMMWeightNoTp,
    NormWeight,
    FusedMoeWeight,
    ROWBMMWeight,
    ROWBMMWeightNoTp,
)
from functools import partial


def fuse_q_kb(self, layer_weight):
    if not (self.weight is None and all(w is not None for w in self.weights)):
        return
    q_weight_ = self.weights[0].transpose(0, 1).contiguous().cpu()
    k_b_proj_ = self.weights[1].contiguous().cpu()
    q_nope_proj_, q_rope_proj_ = torch.split(
        q_weight_.view(-1, layer_weight.tp_q_head_num_, layer_weight.qk_nope_head_dim + layer_weight.qk_rope_head_dim),
        [layer_weight.qk_nope_head_dim, layer_weight.qk_rope_head_dim],
        dim=-1,
    )
    q_nope_proj_ = q_nope_proj_.unsqueeze(2).to(torch.float64)
    k_nope_proj_ = k_b_proj_.unsqueeze(0)
    k_nope_proj_ = k_nope_proj_.to(torch.float64)
    weight = (
        torch.matmul(q_nope_proj_, k_nope_proj_)
        .view(-1, layer_weight.tp_q_head_num_ * layer_weight.kv_lora_rank)
        .transpose(0, 1)
    )
    self.weight = weight.to(self.data_type_)
    self._post_load_weights()


def fuse_vb_o(self, layer_weight):
    if not (self.weight is None and all(w is not None for w in self.weights)):
        return
    v_b_proj_ = self.weights[0].transpose(0, 1)
    o_weight_ = (
        self.weights[1]
        .transpose(0, 1)
        .view(layer_weight.tp_q_head_num_, layer_weight.qk_nope_head_dim, -1)
        .contiguous()
        .to(layer_weight.data_type_)
        .cpu()
    )
    weight = (
        torch.matmul(v_b_proj_.to(torch.float64), o_weight_.to(torch.float64))
        .view(-1, layer_weight.network_config_["hidden_size"])
        .transpose(0, 1)
    )
    self.weight = weight.to(self.data_type_)
    self._post_load_weights()


class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        self.enable_cc_method = os.getenv("ENABLE_CC_METHOD", "False").upper() in ["ON", "TRUE", "1"]
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _parse_config(self):
        super()._parse_config()
        self.is_moe = (
            self.network_config_["n_routed_experts"] is not None
            and self.layer_num_ >= self.network_config_["first_k_dense_replace"]
            and self.layer_num_ % self.network_config_["moe_layer_freq"] == 0
        )
        self.tp_q_head_num_ = self.network_config_["num_attention_heads"]
        if not self.enable_dp:
            self.tp_q_head_num_ = self.tp_q_head_num_ // self.world_size_
        self.n_routed_experts = self.network_config_["n_routed_experts"]
        self.q_lora_rank = self.network_config_["q_lora_rank"]
        self.qk_nope_head_dim = self.network_config_["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.network_config_["qk_rope_head_dim"]
        self.num_attention_heads = self.network_config_["num_attention_heads"]
        self.kv_lora_rank = self.network_config_["kv_lora_rank"]

    def _init_weight_names(self):
        if self.q_lora_rank is None:
            self.rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        else:
            self.rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight"
        self.e_score_correction_bias_name = f"model.layers.{self.layer_num_}.mlp.gate.e_score_correction_bias"

    def _init_qweight_names(self):
        self.act_scale_suffix = None
        self.weight_scale_suffix = None
        if self.quant_cfg.static_activation:
            self.act_scale_suffix = "input_scale"
        if self.quant_cfg.quantized_weight:
            self.weight_scale_suffix = "weight_scale_inv"

    def _init_weight(self):
        if not self.enable_dp:
            self._init_qkvo()
        else:
            self._init_qkvo_dp()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_norm()

    def _load_q_rope(self, q_weight_):
        if not self.enable_dp:
            q_split_n_embed_with_rope = (
                (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads // self.world_size_
            )
            q_weight_ = q_weight_[
                q_split_n_embed_with_rope * self.tp_rank_ : q_split_n_embed_with_rope * (self.tp_rank_ + 1), :
            ]
        q_weight_ = q_weight_.transpose(0, 1).contiguous()
        q_nope_proj_, q_rope_proj_ = torch.split(
            q_weight_.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        return q_rope_proj_.reshape(-1, self.qk_rope_head_dim * self.tp_q_head_num_).transpose(0, 1).contiguous()

    def _load_kb(self, kv_b_proj_):
        k_b_proj_ = kv_b_proj_.view(self.num_attention_heads, self.qk_nope_head_dim * 2, self.kv_lora_rank)[
            :, : self.qk_nope_head_dim, :
        ]
        return k_b_proj_.contiguous().to(kv_b_proj_.dtype)

    def _load_kb_scale(self, kv_b_proj_, block_size):
        k_b_proj_scale_ = kv_b_proj_.view(
            self.num_attention_heads, self.qk_nope_head_dim * 2 // block_size, self.kv_lora_rank // block_size
        )[:, : self.qk_nope_head_dim // block_size, :]
        return k_b_proj_scale_.contiguous().to(kv_b_proj_.dtype)

    def _load_vb(self, kv_b_proj_):
        v_b_proj_ = kv_b_proj_.T.view(self.kv_lora_rank, self.num_attention_heads, self.qk_nope_head_dim * 2,)[
            :, :, self.qk_nope_head_dim :
        ].transpose(0, 1)
        return v_b_proj_.contiguous().to(kv_b_proj_.dtype)

    def _load_vb_scale(self, kv_b_proj_scale_, block_size):
        v_b_proj_scale_ = kv_b_proj_scale_.T.view(
            self.kv_lora_rank // block_size,
            self.num_attention_heads,
            self.qk_nope_head_dim * 2 // block_size,
        )[:, :, self.qk_nope_head_dim // block_size :].transpose(0, 1)
        return v_b_proj_scale_.contiguous().to(kv_b_proj_scale_.dtype)

    def load_hf_weights(self, weights):
        if f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight" in weights:
            kv_b_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight"]
            weights[f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight"] = self._load_kb(kv_b_proj_)
            weights[f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight"] = self._load_vb(kv_b_proj_)
            if self.enable_cc_method:
                weights[f"model.layers.{self.layer_num_}.self_attn.cc_k_b_proj.weight"] = self._load_kb(
                    kv_b_proj_
                ).reshape(-1, self.kv_lora_rank)
                weights[f"model.layers.{self.layer_num_}.self_attn.cc_v_b_proj.weight"] = (
                    self._load_vb(kv_b_proj_)
                    .transpose(0, 1)
                    .reshape(self.kv_lora_rank, -1)
                    .transpose(0, 1)
                    .contiguous()
                )
        if (
            self.quant_cfg.quantized_weight
            and f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + self.weight_scale_suffix in weights
        ):
            kv_b_proj_scale_ = weights[
                f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + self.weight_scale_suffix
            ]
            block_size = 1
            if self.quant_cfg is not None:
                hf_quantization_config = self.quant_cfg.hf_quantization_config
                block_size = hf_quantization_config.get("weight_block_size", [128, 128])[0]
            weights[
                f"model.layers.{self.layer_num_}.self_attn.k_b_proj." + self.weight_scale_suffix
            ] = self._load_kb_scale(kv_b_proj_scale_, block_size)
            weights[
                f"model.layers.{self.layer_num_}.self_attn.v_b_proj." + self.weight_scale_suffix
            ] = self._load_vb_scale(kv_b_proj_scale_, block_size)
            if self.enable_cc_method:
                weights[
                    f"model.layers.{self.layer_num_}.self_attn.cc_k_b_proj." + self.weight_scale_suffix
                ] = self._load_kb_scale(kv_b_proj_scale_, block_size).reshape(-1, self.kv_lora_rank // block_size)
                weights[f"model.layers.{self.layer_num_}.self_attn.cc_v_b_proj." + self.weight_scale_suffix] = (
                    self._load_vb_scale(kv_b_proj_scale_, block_size)
                    .transpose(0, 1)
                    .reshape(self.kv_lora_rank // block_size, -1)
                    .transpose(0, 1)
                    .contiguous()
                )

        return super().load_hf_weights(weights)

    def _set_quantization(self):
        super()._set_quantization()
        # moe_gate of deepseek always keep bf16/fp16.
        if self.is_moe:
            self.moe_gate.quant_method = None

    def _init_qkvo(self):
        q_split_n_embed = self.qk_nope_head_dim * self.tp_q_head_num_
        q_split_n_embed_with_rope = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads // self.world_size_
        )
        if self.q_lora_rank is None:
            self.q_weight_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
        else:
            self.q_a_proj_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight",
                self.data_type_,
                self.q_lora_rank,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.q_b_proj_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

        self.kv_a_proj_with_mqa_ = ROWMMWeightNoTP(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
            self.data_type_,
            self.kv_lora_rank + self.qk_rope_head_dim,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )
        self.k_b_proj_ = ROWBMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )
        self.v_b_proj_ = ROWBMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )
        if self.enable_cc_method:
            self.cc_k_b_proj_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.cc_k_b_proj.weight",
                self.data_type_,
                split_n_embed=self.tp_q_head_num_ * self.qk_nope_head_dim,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.cc_v_b_proj_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.cc_v_b_proj.weight",
                self.data_type_,
                split_n_embed=self.tp_q_head_num_ * self.qk_nope_head_dim,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

        self.o_weight_ = COLMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight",
            self.data_type_,
            q_split_n_embed,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

    def _init_qkvo_dp(self):
        q_split_n_embed = self.qk_nope_head_dim * self.tp_q_head_num_
        q_split_n_embed_with_rope = (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads
        if self.q_lora_rank is None:
            self.q_weight_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
        else:
            self.q_a_proj_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight",
                self.data_type_,
                self.q_lora_rank,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.q_b_proj_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

        self.kv_a_proj_with_mqa_ = ROWMMWeightNoTP(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
            self.data_type_,
            self.kv_lora_rank + self.qk_rope_head_dim,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

        self.k_b_proj_ = ROWBMMWeightNoTp(
            f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

        self.v_b_proj_ = ROWBMMWeightNoTp(
            f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

        self.o_weight_ = COLMMWeightNoTp(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight",
            self.data_type_,
            q_split_n_embed,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

    def _load_mlp(self, mlp_prefix, split_inter_size, no_tp=False):
        if no_tp:
            self.gate_up_proj = MultiROWMMWeightNoTP(
                [f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight"],
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.down_proj = COLMMWeightNoTp(
                f"{mlp_prefix}.down_proj.weight",
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
        else:
            self.gate_up_proj = MultiROWMMWeight(
                [f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight"],
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.down_proj = COLMMWeight(
                f"{mlp_prefix}.down_proj.weight",
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeightNoTP(
            f"model.layers.{self.layer_num_}.mlp.gate.weight",
            self.data_type_,
            moe_intermediate_size,
            weight_scale_suffix=None,
            act_scale_suffix=None,
        )
        shared_intermediate_size = moe_intermediate_size * self.network_config_["n_shared_experts"]
        shared_split_inter_size = shared_intermediate_size // self.world_size_
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts", shared_split_inter_size)

        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            e_score_correction_bias_name=self.e_score_correction_bias_name,
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            split_inter_size=moe_intermediate_size // self.world_size_,
            data_type=self.data_type_,
            network_config=self.network_config_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

    def _init_ffn(self):
        inter_size = self.network_config_["intermediate_size"]
        num_shards = self.world_size_ if not self.enable_dp else 1
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp", inter_size // num_shards, no_tp=self.enable_dp)

    def _init_norm(self):
        self.att_norm_weight_ = NormWeight(f"model.layers.{self.layer_num_}.input_layernorm.weight", self.data_type_)
        self.ffn_norm_weight_ = NormWeight(
            f"model.layers.{self.layer_num_}.post_attention_layernorm.weight", self.data_type_
        )
        self.kv_a_layernorm_ = NormWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_layernorm.weight", self.data_type_
        )
        if self.q_lora_rank is not None:
            self.q_a_layernorm_ = NormWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_a_layernorm.weight", self.data_type_
            )
