import os
import torch
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, FusedMoeWeight


class Glm4MoeLiteTransformerLayerWeight(Deepseek2TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)

    def _parse_config(self):
        # Call parent's _parse_config to set n_embed, moe_inter, and other required attributes
        super()._parse_config()

        # Override is_moe calculation for GLM4 (no moe_layer_freq check)
        self.is_moe = self.network_config_.get(
            "n_routed_experts"
        ) is not None and self.layer_num_ >= self.network_config_.get("first_k_dense_replace", 0)

        # Override num_fused_shared_experts with GLM4-specific logic
        from lightllm.utils.envs_utils import get_env_start_args

        self.num_fused_shared_experts = 0
        if get_env_start_args().enable_fused_shared_experts and self.is_moe:
            moe_mode = os.getenv("MOE_MODE", "TP")
            assert moe_mode == "TP"
            self.num_fused_shared_experts = self.network_config_.get("n_shared_experts", 0)

    def load_hf_weights(self, weights):
        from lightllm.common.basemodel import TransformerLayerWeight
        from lightllm.models.deepseek2.triton_kernel.weight_dequant import weight_dequant

        kv_b_quant_method = self.quant_cfg.get_quant_method(self.layer_num_, "kv_b_proj")
        weight_scale_suffix = None
        if self.quant_cfg.quantized_weight:
            weight_scale_suffix = kv_b_quant_method.weight_scale_suffix

        if f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight" in weights:
            kv_b_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight"]
            # for quantized weights, dequantize first
            if self.quant_cfg.quantized_weight:
                kv_b_proj_ = weight_dequant(
                    kv_b_proj_.cuda(),
                    weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + weight_scale_suffix].cuda(),
                ).cpu()
            # Use GLM4-specific split methods (different from DeepSeek2's dimensions)
            k_b_proj_ = self._load_kb(kv_b_proj_)
            v_b_proj_ = self._load_vb(kv_b_proj_)
            weights[f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight"] = k_b_proj_
            weights[f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight"] = v_b_proj_

        # rename the shared experts weight
        if self.num_fused_shared_experts > 0:
            self._rename_shared_experts(weights, weight_scale_suffix)

        return TransformerLayerWeight.load_hf_weights(self, weights)

    def _load_kb(self, kv_b_proj_):
        kv_dim = self.qk_nope_head_dim + self.v_head_dim
        k_b_proj_ = kv_b_proj_.view(self.num_attention_heads, kv_dim, self.kv_lora_rank)[:, : self.qk_nope_head_dim, :]
        return k_b_proj_.contiguous().to(kv_b_proj_.dtype)

    def _load_kb_scale(self, kv_b_proj_, block_size):
        kv_dim = self.qk_nope_head_dim + self.v_head_dim
        k_b_proj_scale_ = kv_b_proj_.view(
            self.num_attention_heads, kv_dim // block_size, self.kv_lora_rank // block_size
        )[:, : self.qk_nope_head_dim // block_size, :]
        return k_b_proj_scale_.contiguous().to(kv_b_proj_.dtype)

    def _load_vb(self, kv_b_proj_):
        kv_dim = self.qk_nope_head_dim + self.v_head_dim
        v_b_proj_ = kv_b_proj_.T.view(self.kv_lora_rank, self.num_attention_heads, kv_dim)[
            :, :, self.qk_nope_head_dim :
        ].transpose(0, 1)
        return v_b_proj_.contiguous().to(kv_b_proj_.dtype)

    def _load_vb_scale(self, kv_b_proj_scale_, block_size):
        kv_dim = self.qk_nope_head_dim + self.v_head_dim
        v_b_proj_scale_ = kv_b_proj_scale_.T.view(
            self.kv_lora_rank // block_size,
            self.num_attention_heads,
            kv_dim // block_size,
        )[:, :, self.qk_nope_head_dim // block_size :].transpose(0, 1)
        return v_b_proj_scale_.contiguous().to(kv_b_proj_scale_.dtype)

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        hidden_size = self.network_config_["hidden_size"]

        self.moe_gate = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[self.n_routed_experts],
            weight_names=f"model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=torch.float32,  # Router gate needs float32 for numerical stability
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )

        if self.num_fused_shared_experts == 0:
            self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts", is_shared_experts=True)

        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            e_score_correction_bias_name=self.e_score_correction_bias_name,
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )
