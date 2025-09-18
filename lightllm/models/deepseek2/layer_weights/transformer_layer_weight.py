import os
import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    MultiROWMMWeight,
    COLMMWeight,
    NormWeight,
    FusedMoeWeightTP,
    FusedMoeWeightEP,
    ROWBMMWeight,
)
from functools import partial
from ..triton_kernel.weight_dequant import weight_dequant


class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        self.enable_cc_method = not os.getenv("DISABLE_CC_METHOD", "False").upper() in ["ON", "TRUE", "1"]
        super().__init__(layer_num, data_type, network_config, mode, quant_cfg)
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
            # MOE_MODE 处于 TP 模式下才能使能 enable_fused_shared_experts
            moe_mode = os.getenv("MOE_MODE", "TP")
            assert moe_mode == "TP"
            self.num_fused_shared_experts = self.network_config_.get("n_shared_experts", 0)

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
            weights[f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight"] = self._load_kb(kv_b_proj_)
            weights[f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight"] = self._load_vb(kv_b_proj_)

        if (
            self.quant_cfg.quantized_weight
            and f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + weight_scale_suffix in weights
        ):
            kv_b_proj_scale_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + weight_scale_suffix]
            block_size = 128
            weights[f"model.layers.{self.layer_num_}.self_attn.k_b_proj." + weight_scale_suffix] = self._load_kb_scale(
                kv_b_proj_scale_, block_size
            )
            weights[f"model.layers.{self.layer_num_}.self_attn.v_b_proj." + weight_scale_suffix] = self._load_vb_scale(
                kv_b_proj_scale_, block_size
            )

        # rename the shared experts weight
        if self.num_fused_shared_experts > 0:
            self._rename_shared_experts(weights, weight_scale_suffix)
        return super().load_hf_weights(weights)

    def _init_qkvo(self):
        if self.q_lora_rank is None:
            self.q_weight_ = ROWMMWeight(
                weight_name=f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="q_weight",
            )
            self.kv_a_proj_with_mqa_ = ROWMMWeight(
                weight_name=f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="kv_a_proj_with_mqa",
                tp_rank=0,
                tp_world_size=1,
            )
        else:
            self.qkv_a_proj_with_mqa_ = MultiROWMMWeight(
                weight_names=[
                    f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight",
                    f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
                ],
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="qkv_a_proj_with_mqa",
                tp_rank=0,
                tp_world_size=1,
            )
            self.q_b_proj_ = ROWMMWeight(
                weight_name=f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="q_b_proj",
            )
        self.k_b_proj_ = ROWBMMWeight(
            weight_name=f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight",
            data_type=self.data_type_,
            quant_cfg=None,
            layer_num=self.layer_num_,
            name="k_b_proj",
        )
        self.v_b_proj_ = ROWBMMWeight(
            weight_name=f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight",
            data_type=self.data_type_,
            quant_cfg=None,
            layer_num=self.layer_num_,
            name="v_b_proj",
        )
        if self.enable_cc_method:
            self.cc_kv_b_proj_ = ROWMMWeight(
                weight_name=f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight",
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="cc_kv_b_proj",
            )

        self.o_weight_ = COLMMWeight(
            weight_name=f"model.layers.{self.layer_num_}.self_attn.o_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="o_weight",
        )

    def _load_mlp(self, mlp_prefix):
        self.gate_up_proj = MultiROWMMWeight(
            weight_names=[f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="gate_up_proj",
        )
        self.down_proj = COLMMWeight(
            weight_name=f"{mlp_prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="down_proj",
        )

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            weight_name=f"model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=self.data_type_,
            layer_num=self.layer_num_,
            name="moe_gate",
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
            self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts")
        moe_mode = os.getenv("MOE_MODE", "TP")
        assert moe_mode in ["EP", "TP"]
        if moe_mode == "TP":
            self.experts = FusedMoeWeightTP(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name=self.e_score_correction_bias_name,
                weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                split_inter_size=moe_intermediate_size // self.tp_world_size_,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
            )
        elif moe_mode == "EP":
            self.experts = FusedMoeWeightEP(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name=self.e_score_correction_bias_name,
                weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
            )
        else:
            raise ValueError(f"Unsupported moe mode: {moe_mode}")

    def _init_ffn(self):
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp")

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
