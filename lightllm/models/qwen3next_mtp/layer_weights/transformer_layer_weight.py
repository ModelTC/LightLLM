import os
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    NormWeight,
    FusedMoeWeightEP,
    create_tp_moe_wegiht_obj,
)
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight


class Qwen3NextMTPTransformerLayerWeight(Qwen3MOETransformerLayerWeight):
    """
    Qwen3Next MTP Transformer Layer Weight.
    MTP layers use 'mtp.layers.{layer_num}' prefix instead of 'model.layers.{layer_num}'.
    MTP layers are always full attention (not linear attention) with MoE FFN.
    """

    def __init__(self, layer_num, data_type, network_config, mode=[], quant_cfg=None):
        # MTP always uses MoE
        self.n_routed_experts = network_config["num_experts"]
        self.is_moe = True
        super(Qwen3MOETransformerLayerWeight, self).__init__(layer_num, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight_names(self):
        # Override weight names to use 'mtp.layers' prefix
        self._q_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_norm_name = f"mtp.layers.{self.layer_num_}.self_attn.q_norm.weight"
        self._q_bias_name = None
        self._k_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_norm_name = f"mtp.layers.{self.layer_num_}.self_attn.k_norm.weight"
        self._k_bias_name = None
        self._v_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._kv_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.kv_proj.weight"
        self._kv_bias_name = None
        self._o_weight_name = f"mtp.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None
        self._att_norm_weight_name = f"mtp.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"mtp.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        self._init_moe()
        self._init_norm()
        self._init_shared_expert_weight()

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            weight_names=f"mtp.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=self.data_type_,
            layer_num=self.layer_num_,
            name="moe_gate",
            tp_rank=0,
            tp_world_size=1,
        )
        moe_mode = os.getenv("MOE_MODE", "TP")
        assert moe_mode in ["EP", "TP"]
        if moe_mode == "TP":
            self.experts = create_tp_moe_wegiht_obj(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name="",
                weight_prefix=f"mtp.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                split_inter_size=moe_intermediate_size // self.tp_world_size_,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
                num_fused_shared_experts=0,
            )
        elif moe_mode == "EP":
            self.experts = FusedMoeWeightEP(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
                e_score_correction_bias_name="",
                weight_prefix=f"mtp.layers.{self.layer_num_}.mlp.experts",
                n_routed_experts=self.n_routed_experts,
                data_type=self.data_type_,
                network_config=self.network_config_,
                layer_num=self.layer_num_,
                quant_cfg=self.quant_cfg,
            )
        else:
            raise ValueError(f"Unsupported moe mode: {moe_mode}")

    def _init_shared_expert_weight(self):
        prefix = f"mtp.layers.{self.layer_num_}.mlp.shared_expert"
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
            weight_names=f"mtp.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate",
            tp_rank=0,
            tp_world_size=1,
        )
