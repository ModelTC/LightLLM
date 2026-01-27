import os
import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    ROWBMMWeight,
    COLMMWeight,
    RMSNormWeight,
    FusedMoeWeight,
)
from ..triton_kernel.weight_dequant import weight_dequant


class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        self.enable_cc_method = not os.getenv("DISABLE_CC_METHOD", "False").upper() in ["ON", "TRUE", "1"]
        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _parse_config(self):
        super()._parse_config()
        self.is_moe = (
            self.network_config_["n_routed_experts"] is not None
            and self.layer_num_ >= self.network_config_["first_k_dense_replace"]
            and self.layer_num_ % self.network_config_["moe_layer_freq"] == 0
        )
        self.tp_q_head_num_ = self.network_config_["num_attention_heads"]
        self.tp_q_head_num_ = self.tp_q_head_num_ // self.tp_world_size_
        self.n_routed_experts = self.network_config_["n_routed_experts"]
        self.q_lora_rank = self.network_config_["q_lora_rank"]
        self.qk_nope_head_dim = self.network_config_["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.network_config_["qk_rope_head_dim"]
        self.v_head_dim = self.network_config_["v_head_dim"]
        self.num_attention_heads = self.network_config_["num_attention_heads"]
        self.kv_lora_rank = self.network_config_["kv_lora_rank"]
        self.num_fused_shared_experts = 0
        if get_env_start_args().enable_fused_shared_experts and self.is_moe:
            # enable_fused_shared_experts can only work with tensor parallelism
            assert not get_env_start_args().enable_ep_moe, "enable_fused_shared_experts can only work with tp mode."
            self.num_fused_shared_experts = self.network_config_.get("n_shared_experts", 0)
        self.n_embed = self.network_config_["hidden_size"]
        self.n_inter = self.network_config_["intermediate_size"]
        self.moe_inter = self.network_config_.get("moe_intermediate_size", self.n_inter)
        self.q_out_dim = self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        self.kv_a_out_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.kv_b_out_dim = self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.o_in_dim = self.num_attention_heads * self.v_head_dim

    def _init_weight_names(self):
        if self.q_lora_rank is None:
            self.rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        else:
            self.rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight"
        self.e_score_correction_bias_name = f"model.layers.{self.layer_num_}.mlp.gate.e_score_correction_bias"

    def _init_weight(self):
        self._init_qkvo()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_norm()

    def _split_kv_b_proj(self, kv_b_proj_):
        kv_b_proj_ = kv_b_proj_.view(self.num_attention_heads, self.qk_nope_head_dim * 2, self.kv_lora_rank)
        k_b_proj_, v_b_proj_ = torch.split(kv_b_proj_, [self.qk_nope_head_dim, self.v_head_dim], dim=-2)
        # num_attention_heads x qk_nope_head_dim x kv_lora_rank
        k_b_proj_ = k_b_proj_.contiguous().to(kv_b_proj_.dtype)
        # num_attention_heads x kv_lora_rank x v_head_dim
        v_b_proj_ = v_b_proj_.transpose(1, 2).contiguous().to(kv_b_proj_.dtype)
        return k_b_proj_, v_b_proj_

    def _rename_shared_experts(self, weights, weight_scale_suffix):
        # 将共享专家对应的参数，改造为与路由专家一致的权重名称和映射关系。
        old_prefix = f"model.layers.{self.layer_num_}.mlp.shared_experts"
        new_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        proj_names = ["gate_proj", "down_proj", "up_proj"]
        for i in range(self.num_fused_shared_experts):
            expert_id = self.n_routed_experts + i
            for proj in proj_names:
                weight_tensor = weights.get(f"{old_prefix}.{proj}.weight")
                if weight_tensor is not None:
                    weights[f"{new_prefix}.{expert_id}.{proj}.weight"] = weight_tensor
                if self.quant_cfg.quantized_weight:
                    assert weight_scale_suffix is not None
                    scale_tensor = weights.get(f"{old_prefix}.{proj}." + weight_scale_suffix)
                    if scale_tensor is not None:
                        weights[f"{new_prefix}.{expert_id}.{proj}." + weight_scale_suffix] = scale_tensor

    def load_hf_weights(self, weights):
        kv_b_quant_method = self.quant_cfg.get_quant_method(self.layer_num_, "kv_b_proj")
        weight_scale_suffix = None
        if self.quant_cfg.quantized_weight:
            weight_scale_suffix = kv_b_quant_method.weight_scale_suffix
        if f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight" in weights:
            kv_b_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight"]
            # for deepseek_v3, the bmm operator is not quantized
            if self.quant_cfg.quantized_weight:
                kv_b_proj_ = weight_dequant(
                    kv_b_proj_.cuda(),
                    weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + weight_scale_suffix].cuda(),
                ).cpu()
            k_b_proj_, v_b_proj_ = self._split_kv_b_proj(kv_b_proj_)
            weights[f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight"] = k_b_proj_
            weights[f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight"] = v_b_proj_

        # rename the shared experts weight
        if self.num_fused_shared_experts > 0:
            self._rename_shared_experts(weights, weight_scale_suffix)
        return super().load_hf_weights(weights)

    def _init_qkvo(self):
        if self.q_lora_rank is None:
            self.q_weight_ = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[self.q_out_dim],
                weight_names=f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("q_weight"),
            )
            self.kv_a_proj_with_mqa_ = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[self.kv_a_out_dim],
                weight_names=f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("kv_a_proj_with_mqa"),
                tp_rank=0,
                tp_world_size=1,
            )
        else:
            self.qkv_a_proj_with_mqa_ = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[self.q_lora_rank, self.kv_a_out_dim],
                weight_names=[
                    f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight",
                    f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
                ],
                data_type=self.data_type_,
                quant_method=self.get_quant_method("qkv_a_proj_with_mqa"),
                tp_rank=0,
                tp_world_size=1,
            )
            self.q_b_proj_ = ROWMMWeight(
                in_dim=self.q_lora_rank,
                out_dims=[self.q_out_dim],
                weight_names=f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("q_b_proj"),
            )
        self.k_b_proj_ = ROWBMMWeight(
            dim0=self.num_attention_heads,
            dim1=self.qk_nope_head_dim,
            dim2=self.kv_lora_rank,
            weight_names=f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight",
            data_type=self.data_type_,
            quant_method=None,
        )
        self.v_b_proj_ = ROWBMMWeight(
            dim0=self.num_attention_heads,
            dim1=self.kv_lora_rank,
            dim2=self.v_head_dim,
            weight_names=f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight",
            data_type=self.data_type_,
            quant_method=None,
        )
        if self.enable_cc_method:
            self.cc_kv_b_proj_ = ROWMMWeight(
                in_dim=self.kv_lora_rank,
                out_dims=[self.kv_b_out_dim],
                weight_names=f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("cc_kv_b_proj"),
            )

        self.o_weight_ = COLMMWeight(
            in_dim=self.o_in_dim,
            out_dims=[self.n_embed],
            weight_names=f"model.layers.{self.layer_num_}.self_attn.o_proj.weight",
            data_type=self.data_type_,
            quant_method=self.get_quant_method("o_weight"),
        )

    def _load_mlp(self, mlp_prefix, is_shared_experts=False):
        enable_ep_moe = get_env_start_args().enable_ep_moe
        mlp_inter = self.moe_inter if is_shared_experts else self.n_inter
        if self.is_moe and enable_ep_moe:
            self.gate_up_proj = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[mlp_inter, mlp_inter],
                weight_names=[f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight"],
                data_type=self.data_type_,
                quant_method=self.get_quant_method("gate_up_proj"),
                tp_rank=0,
                tp_world_size=1,
            )
            self.down_proj = COLMMWeight(
                in_dim=mlp_inter,
                out_dims=[self.n_embed],
                weight_names=f"{mlp_prefix}.down_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("down_proj"),
                tp_rank=0,
                tp_world_size=1,
            )
        else:
            self.gate_up_proj = ROWMMWeight(
                in_dim=self.n_embed,
                out_dims=[mlp_inter, mlp_inter],
                weight_names=[f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight"],
                data_type=self.data_type_,
                quant_method=self.get_quant_method("gate_up_proj"),
            )
            self.down_proj = COLMMWeight(
                in_dim=mlp_inter,
                out_dims=[self.n_embed],
                weight_names=f"{mlp_prefix}.down_proj.weight",
                data_type=self.data_type_,
                quant_method=self.get_quant_method("down_proj"),
            )

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            in_dim=self.n_embed,
            out_dims=[self.n_routed_experts],
            weight_names=f"model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        # deepseekv3 模型初始几层是非moe架构，后续层才是moe架构
        # 当使能了共享专家融合策略时，共享专家不再以普通的mlp形式进行
        # 加载，而是和路由专家一起融合成一体进行推理，所以当发现当前
        # 层是moe，同时使能了共享专家融合功能时，不初始化独立的共享
        # 专家对应的 gate_up_proj 等weight 参数。当 num_fused_shared_experts
        # == 0 时，说明不存在融合共享专家，共享专家单独加载和进行推理。
        if self.num_fused_shared_experts == 0:
            self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts", is_shared_experts=True)
        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            e_score_correction_bias_name=self.e_score_correction_bias_name,
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=self.n_embed,
            moe_intermediate_size=moe_intermediate_size,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )

    def _init_ffn(self):
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp")

    def _init_norm(self):
        hidden_size = self.network_config_["hidden_size"]

        self.att_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"model.layers.{self.layer_num_}.input_layernorm.weight",
            data_type=self.data_type_,
        )
        self.ffn_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name=f"model.layers.{self.layer_num_}.post_attention_layernorm.weight",
            data_type=self.data_type_,
        )
        self.kv_a_layernorm_ = RMSNormWeight(
            dim=self.kv_lora_rank,
            weight_name=f"model.layers.{self.layer_num_}.self_attn.kv_a_layernorm.weight",
            data_type=self.data_type_,
        )
        if self.q_lora_rank is not None:
            self.q_a_layernorm_ = RMSNormWeight(
                dim=self.q_lora_rank,
                weight_name=f"model.layers.{self.layer_num_}.self_attn.q_a_layernorm.weight",
                data_type=self.data_type_,
            )
