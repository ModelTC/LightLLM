import os
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    NoTpNormWeight,
    FusedMoeWeightEP,
    create_tp_moe_wegiht_obj,
)
from lightllm.utils.envs_utils import get_env_start_args


class Glm4MoeTransformerLayerWeight(LlamaTransformerLayerWeight):
    """
    GLM-4.7 MoE Transformer Layer Weight.

    GLM-4.7 architecture:
    - 160 routed experts + 1 shared expert
    - Top-8 expert routing per token
    - Sigmoid gating with e_score_correction_bias
    - routed_scaling_factor = 2.5
    - First 3 layers are dense (first_k_dense_replace=3)
    - QK normalization (like Qwen3)
    - Partial rotary embeddings (0.5 factor)
    """

    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        # Parse MoE-specific config before calling super().__init__
        self.n_routed_experts = network_config.get("n_routed_experts", 160)
        self.n_shared_experts = network_config.get("n_shared_experts", 1)
        first_k_dense_replace = network_config.get("first_k_dense_replace", 3)

        # Determine if this layer is MoE (layers >= first_k_dense_replace are MoE)
        self.is_moe = self.n_routed_experts is not None and layer_num >= first_k_dense_replace

        # Check if fused shared experts is enabled (only works in TP mode)
        self.num_fused_shared_experts = 0
        if get_env_start_args().enable_fused_shared_experts and self.is_moe:
            moe_mode = os.getenv("MOE_MODE", "TP")
            if moe_mode == "TP":
                self.num_fused_shared_experts = self.n_shared_experts

        super().__init__(layer_num, data_type, network_config, quant_cfg)
        return

    def _parse_config(self):
        super()._parse_config()
        # GLM-4.7 specific config
        self.moe_intermediate_size = self.network_config_.get("moe_intermediate_size", 1536)
        self.routed_scaling_factor = self.network_config_.get("routed_scaling_factor", 2.5)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        # GLM-4.7 has attention biases on q/k/v projections (but not o_proj)
        self._q_bias_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"
        self._k_bias_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"
        self._v_bias_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"
        # QK normalization weight names (like Qwen3)
        self._q_norm_name = f"model.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._k_norm_name = f"model.layers.{self.layer_num_}.self_attn.k_norm.weight"
        # e_score_correction_bias for MoE routing
        self._e_score_correction_bias_name = f"model.layers.{self.layer_num_}.mlp.gate.e_score_correction_bias"
        return

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_norm()
        return

    def _init_qkv(self):
        """Initialize QKV projections - same as parent but with QK norm weights."""
        super()._init_qkv()
        return

    def _init_norm(self):
        """Initialize normalization weights including QK norms."""
        super()._init_norm()
        # Add QK normalization weights (like Qwen3)
        self.q_norm_weight_ = NoTpNormWeight(weight_name=self._q_norm_name, data_type=self.data_type_)
        self.k_norm_weight_ = NoTpNormWeight(weight_name=self._k_norm_name, data_type=self.data_type_)
        return

    def _init_moe(self):
        """Initialize MoE weights (gate, routed experts, shared experts)."""
        # MoE gate for routing
        self.moe_gate = ROWMMWeight(
            weight_names=f"model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=self.data_type_,
            layer_num=self.layer_num_,
            name="moe_gate",
            tp_rank=0,
            tp_world_size=1,
        )

        # If fused shared experts is not enabled, load shared experts separately
        if self.num_fused_shared_experts == 0 and self.n_shared_experts is not None and self.n_shared_experts > 0:
            self._load_shared_experts()

        # Initialize routed experts based on parallelism mode
        moe_mode = os.getenv("MOE_MODE", "TP")
        assert moe_mode in ["EP", "TP"], f"Unsupported MOE_MODE: {moe_mode}"

        if moe_mode == "TP":
            self.experts = create_tp_moe_wegiht_obj(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name=self._e_score_correction_bias_name,
                weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                split_inter_size=self.moe_intermediate_size // self.tp_world_size_,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
            )
        else:  # EP mode
            self.experts = FusedMoeWeightEP(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name=self._e_score_correction_bias_name,
                weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
            )
        return

    def _load_shared_experts(self):
        """Load shared expert weights as standard FFN (gate_up_proj + down_proj)."""
        shared_prefix = f"model.layers.{self.layer_num_}.mlp.shared_experts"
        moe_mode = os.getenv("MOE_MODE", "TP")

        if moe_mode == "EP":
            # In EP mode, shared experts don't use TP splitting
            self.gate_up_proj = ROWMMWeight(
                weight_names=[f"{shared_prefix}.gate_proj.weight", f"{shared_prefix}.up_proj.weight"],
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="shared_gate_up_proj",
                tp_rank=0,
                tp_world_size=1,
            )
            self.down_proj = COLMMWeight(
                weight_names=f"{shared_prefix}.down_proj.weight",
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="shared_down_proj",
                tp_rank=0,
                tp_world_size=1,
            )
        else:
            # In TP mode, shared experts use TP splitting
            self.gate_up_proj = ROWMMWeight(
                weight_names=[f"{shared_prefix}.gate_proj.weight", f"{shared_prefix}.up_proj.weight"],
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="shared_gate_up_proj",
            )
            self.down_proj = COLMMWeight(
                weight_names=f"{shared_prefix}.down_proj.weight",
                data_type=self.data_type_,
                quant_cfg=self.quant_cfg,
                layer_num=self.layer_num_,
                name="shared_down_proj",
            )
        return

    def _rename_shared_experts(self, weights, weight_scale_suffix):
        """Rename shared expert weights to match routed expert naming for fusion."""
        old_prefix = f"model.layers.{self.layer_num_}.mlp.shared_experts"
        new_prefix = f"model.layers.{self.layer_num_}.mlp.experts"
        proj_names = ["gate_proj", "down_proj", "up_proj"]

        for i in range(self.num_fused_shared_experts):
            expert_id = self.n_routed_experts + i
            for proj in proj_names:
                weight_tensor = weights.get(f"{old_prefix}.{proj}.weight")
                if weight_tensor is not None:
                    weights[f"{new_prefix}.{expert_id}.{proj}.weight"] = weight_tensor
                if self.quant_cfg.quantized_weight and weight_scale_suffix:
                    scale_tensor = weights.get(f"{old_prefix}.{proj}." + weight_scale_suffix)
                    if scale_tensor is not None:
                        weights[f"{new_prefix}.{expert_id}.{proj}." + weight_scale_suffix] = scale_tensor
        return

    def load_hf_weights(self, weights):
        """Load weights from HuggingFace format, handling shared expert fusion if enabled."""
        # Handle shared expert renaming for fusion
        if self.num_fused_shared_experts > 0:
            weight_scale_suffix = None
            if self.quant_cfg.quantized_weight:
                kv_b_quant_method = self.quant_cfg.get_quant_method(self.layer_num_, "kv_b_proj")
                weight_scale_suffix = kv_b_quant_method.weight_scale_suffix
            self._rename_shared_experts(weights, weight_scale_suffix)

        return super().load_hf_weights(weights)
